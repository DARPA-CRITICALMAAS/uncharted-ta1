from google.cloud.vision import ImageAnnotatorClient
from google.cloud.vision import Image as VisionImage
from google.cloud.vision_v1.types.geometry import BoundingPoly, Vertex
from PIL.Image import Image as PILImage
import cv2 as cv
import numpy as np
import io, copy, logging
from typing import List, Dict, Any, Optional, Tuple
from shapely.geometry import Polygon
from shapely import MultiPolygon, unary_union, concave_hull
from tasks.text_extraction.ocr.cloud_authenticator import CloudAuthenticator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GoogleVisionOCR:
    """
    Extracts text from an image using Google Vision's OCR API
    """

    def __init__(self, ocr_cloud_auth=False):
        """
        Initializes the GoogleVisionOCR class.

        Args:
            ocr_cloud_auth (bool): If True, authenticate using cloud credentials and initialize
                                       the Vision API client with cloud authentication. If False,
                                       initialize the Vision API client with default credentials.
        """
        if ocr_cloud_auth:
            logger.info("Authenticating with cloud credentials")
            cloud_authenticator = CloudAuthenticator()
            self._client = cloud_authenticator.get_vision_client()
        else:
            logger.info("Authenticating with default credentials")
            self._client = ImageAnnotatorClient()

    def validate_api_key(self) -> None:
        """
        Validates the API key for the Google Vision OCR client.
        This method checks if the API key is valid by attempting to make a batch
        annotation request with an empty list of requests. If the API key is invalid,
        an exception will be raised.
        Raises:
            google.api_core.exceptions.GoogleAPIError: If the API key is invalid or
            there is an issue with the request.
        """
        self._client.batch_annotate_images(requests=[])
        logger.debug("API key validated")

    def detect_text(self, vision_img: VisionImage) -> List[Dict[str, Any]]:
        """
        Runs google vision OCR on an image and returns the text annotations as a dictionary

        Args:
            vision_img: input image (google vision Image class)

        Returns:
            List of text extractions -- {"text": <extracted text>, "bounding_poly": <google BoundingPoly object>, "confidence": <extraction confidence>}
        """
        # (from https://cloud.google.com/vision/docs/ocr#vision_text_detection-python)
        # TODO -- Note: extraction 'score' always seems to be 0.0 ??

        response = self._client.text_detection(image=vision_img)  # type: ignore

        text_extractions: List[Dict[str, Any]] = []
        if response.text_annotations:
            # note: first entry will be the entire text block
            text_extractions = [
                {
                    "text": text.description,
                    "bounding_poly": text.bounding_poly,
                    "confidence": text.score,
                }  # text.confidence is deprecated
                for text in response.text_annotations
            ]
        else:
            logger.warning("No OCR text found!")

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )

        return text_extractions

    def _build_text_block(
        self, text_block: str, text_block_next: str, texts_subset: List[Dict[str, Any]]
    ) -> Tuple[Optional[BoundingPoly], int]:
        group_counter = 0
        bounding_poly = None
        try:
            for text in texts_subset:
                prose = text["text"].strip()
                group_counter += 1
                text_block_sub = text_block.replace(
                    prose, "", 1
                ).strip()  # TODO could make this replace from the start of string...
                if len(text_block_sub) == 0:
                    bounding_poly = self._add_bounding_polygons(
                        bounding_poly, text["bounding_poly"]
                    )
                    break
                elif (
                    len(prose) > 0
                    and len(text_block_sub) == len(text_block)
                    and text_block_next.startswith(prose)
                ):  # and len(text_block_sub) < 3
                    # partial OCR block parsed! ... finish with this block and go to next one
                    group_counter -= 1
                    break
                bounding_poly = self._add_bounding_polygons(
                    bounding_poly, text["bounding_poly"]
                )
                text_block = text_block_sub
        except:
            logger.error("error joining ocr blocks")
        return bounding_poly, group_counter

    def text_to_blocks(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean up extract OCR text list into blocks of continuus text (lines)
        and adjust bounding-boxes (polygons) as needed
        NOTE:
        - meant to be used in conjunction with `detect_text` func
        - assumes first element contains the full text delimited by line breaks)
        """

        if len(texts) < 2:
            logger.warning(
                "Less than 2 blocks of OCR text found! Skipping ocr_text_to_blocks"
            )
            return []

        full_text = texts[0]["text"]

        group_offset = 1
        num_blocks = 0

        results: List[Dict[str, Any]] = []

        text_blocks = full_text.split("\n")
        text_block = text_blocks[0]
        bounding_poly: Optional[BoundingPoly] = None
        for text_block_next in text_blocks[1:]:
            text_block = text_block.strip()
            text_block0 = text_block
            bounding_poly, group_counter = self._build_text_block(
                text_block, text_block_next, texts[group_offset:]
            )
            num_blocks += 1

            # TODO could try this too.. for bounding_poly
            # from google.protobuf.json_format import MessageToDict
            # dict_obj = MessageToDict(org)
            if bounding_poly is not None:
                results.append({"text": text_block0, "bounding_poly": bounding_poly})
                bounding_poly = None
            group_offset += group_counter
            text_block = text_block_next

        # and save last block too
        bounding_poly, _ = self._build_text_block(text_block, "", texts[group_offset:])
        if bounding_poly is not None:
            results.append({"text": text_block.strip(), "bounding_poly": bounding_poly})
        if len(results) < len(text_blocks) - 1:
            logger.warning(
                "Possible error grouping OCR results"
            )  # TODO - throw exception here?

        return results

    def detect_document_text(self, vision_img: VisionImage) -> List[Dict[str, Any]]:
        """
        Runs google vision OCR on the image and returns the text annotations as a dictionary
        """
        response = self._client.document_text_detection(image=vision_img)  # type: ignore

        text_extractions = []
        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_words = []
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join(
                                [symbol.text for symbol in word.symbols]
                            )
                            block_words.append(word_text)

                    text_extractions.append(
                        {
                            "text": " ".join(block_words),
                            "bounding_poly": block.bounding_box,
                            "confidence": block.confidence,
                        }
                    )
        else:
            logger.warning("No OCR text found!")

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )

        return text_extractions

    def _add_bounding_polygons(
        self, poly1: Optional[BoundingPoly], poly2: Optional[BoundingPoly]
    ) -> Optional[BoundingPoly]:
        """
        Adds two google vision 'bounding polygon' objects together to create a new bounding polygon that contains both
        """

        if poly1 is None:
            return poly2
        if poly2 is None:
            return poly1

        # use Shapely to combine polygons together
        p1_coords = [(v.x, v.y) for v in poly1.vertices]  # type: ignore
        p1 = Polygon(p1_coords)
        p2_coords = [(v.x, v.y) for v in poly2.vertices]  # type: ignore
        p2 = Polygon(p2_coords)
        p_union = unary_union([p1, p2])

        if isinstance(p_union, MultiPolygon):
            # input polygons don't overlap, so merge into one single 'parent' polygon using concave hull
            p_union = concave_hull(p_union, ratio=1)

        # remove any redundant vertices in final polygon
        p_union = p_union.simplify(0.0)

        if not isinstance(p_union, Polygon):
            logger.error(
                f"Error combining bounding polygons - result is {type(p_union)}"
            )
            return None

        # convert shapely polygon result back to Google Vision BoundingPoly object
        verts = [
            Vertex({"x": int(x), "y": int(y)})
            for x, y in zip(p_union.exterior.xy[0], p_union.exterior.xy[1])
        ]
        if verts[0].x == verts[-1].x and verts[0].y == verts[-1].y:
            # first and last vertices are duplicates - so, ok to remove last vertex (redundant)
            verts.pop()
        poly_combined = BoundingPoly({"vertices": verts})

        return poly_combined

    @staticmethod
    def scale_ocr_coords(
        text_blocks: List[Dict[str, Any]], resize_factor: float
    ) -> List[Dict[str, Any]]:
        """
        scale ocr pixel coords by a re-size factor
        """

        if not text_blocks or resize_factor == 1.0:
            return text_blocks
        # TODO could add a check that any bounding_poly coords are within original image boundaries
        for blk in text_blocks:
            for v in blk["bounding_poly"].vertices:
                v.x = int(v.x * resize_factor)
                v.y = int(v.y * resize_factor)

        return text_blocks

    @staticmethod
    def offset_ocr_coords(
        text_blocks: List[Dict[str, Any]], offset: tuple
    ) -> List[Dict[str, Any]]:
        """
        offset ocr pixel coords by (x,y) co-ords
        """

        if not text_blocks or not offset:
            return text_blocks
        for blk in text_blocks:
            for v in blk["bounding_poly"].vertices:
                v.x = int(v.x + offset[0])
                v.y = int(v.y + offset[1])

        return text_blocks

    @staticmethod
    def bounding_polygon_to_list(bounding_poly: BoundingPoly) -> List[Tuple[int, int]]:
        """
        convert google vision BoundingPoly object to list of x,y points
        """
        poly_pts = [(vertex.x, vertex.y) for vertex in bounding_poly.vertices]  # type: ignore

        return poly_pts

    # TODO -- some of these static methods could go into a 'common' module??
    # E.g., esp the load_* image functions?

    @staticmethod
    def load_vision_image(path: str) -> VisionImage:
        """
        Loads an image into memory as a google vision api image object
        """
        with io.open(path, "rb") as image_file:
            content = image_file.read()
        return VisionImage(content=content)

    @staticmethod
    def pil_to_vision_image(pil_image: PILImage) -> VisionImage:
        """
        Converts a PIL image object to a google vision api image object
        """
        pil_image_bytes = io.BytesIO()
        pil_image.save(pil_image_bytes, format="JPEG")
        return VisionImage(content=pil_image_bytes.getvalue())

    @staticmethod
    def cv_to_vision_image(cv_image: np.ndarray) -> VisionImage:
        """
        Converts an opencv image (numpy array) to a google vision api image object
        """
        # (from https://jdhao.github.io/2019/07/06/python_opencv_pil_image_to_bytes/)
        success, encoded_image = cv.imencode(".jpg", cv_image)
        return VisionImage(content=encoded_image.tobytes())
