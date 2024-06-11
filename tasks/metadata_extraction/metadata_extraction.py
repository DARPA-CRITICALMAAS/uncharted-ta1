import copy
import logging
import os
import re
import json
from enum import Enum
from unittest.mock import Base
from langchain_openai import ChatOpenAI
import cv2
import numpy as np
from PIL.Image import Image as PILImage
from langchain.schema import SystemMessage, PromptValue
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
import tiktoken
from tasks.common.image_io import pil_to_cv_image
from tasks.common.task import TaskInput, TaskResult
from tasks.metadata_extraction.entities import (
    MapChromaType,
    MapShape,
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.segmentation.entities import SEGMENTATION_OUTPUT_KEY, MapSegmentation
from tasks.text_extraction.entities import (
    DocTextExtraction,
    TextExtraction,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.common.pipeline import Task
from typing import Callable, List, Dict, Any, Optional, Tuple


logger = logging.getLogger("metadata_extractor")

PLACE_EXTENSION_MAP = {"washington": "washington (state)"}


class LLM(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_4_O = "gpt-4o"

    def __str__(self):
        return self.value


class MetdataLLM(BaseModel):
    title: str = Field(
        description="The title of the map. If this includes state names, "
        + "county names or quadrangles, still include them in full title. "
        + " Example: 'Geologic map of the Grand Canyon Quadrangle, Arizona'"
    )
    authors: List[str] = Field(
        description="The authors of the map. "
        + "Should be a list of strings in the format <last name, first iniital, middle initial>"
        + "Example of author name: 'Bailey, D. K.'"
        + "References, citations and geology attribution should be ignored when extracting authors."
        + "A single author is allowed. Authors, title and year are normally grouped together."
    )
    year: str = Field(
        description="The year the map was published"
        + "Should be a single 4 digit number and the most recent year if multiple are present"
    )
    scale: str = Field(description="The scale of the map.  Example: '1:24000'")
    datum: str = Field(
        description="The datum of the map."
        + "Examples: 'North American Datum of 1927', 'NAD83', 'WGS 84'"
    )
    vertical_datum: str = Field(
        description="The vertical datum of the map."
        + "Examples: 'mean sea level', 'vertical datum of 1901', 'national vertical geoditic datum of 1929'"
    )
    projection: str = Field(
        description="The map projection."
        + "Examples: 'Polyconic', 'Lambert', 'Transverse Mercator'"
    )
    coordinate_systems: List[str] = Field(
        description="The coordinate systems present on the map."
        + "Examples: 'Utah coordinate system central zone', 'UTM Zone 15', "
        + "'Universal Transverse Mercator zone 12', "
        + "'New Mexico coordinate system, north zone', 'Carter Coordinate System'."
        + "The term `grid ticks` should not be included in coordinate system output."
    )
    base_map: str = Field(
        "The base map information description.  The base map description can be a "
        + "descriptive string, but also often contains (quadrangle, year) pairs."
        + "Examples: 'U.S. Geological Survey 1954', "
        + "'U.S. Geological Survey 1:62,500', "
        + "'Vidal (1949) Rice and Turtle Mountains (1954) Savahia Peak (1975)'"
    )
    counties: List[str] = Field(
        description="Counties covered by the map.  These are often listed in the title; "
        + "if they are not, they should be extracted from the map."
    )
    states: List[str] = Field(
        description="States or provinces covered by the map.  States "
        + "includes principal subvidisions of any country and their full "
        + "name should be extracted. Examples: 'Arizona', 'New York',"
        + " 'South Dakota', 'Ontario'"
    )
    country: str = Field(
        description="Country covered by the map." + "Examples: 'USA', 'Canada'"
    )
    publisher: str = Field(description="The publisher of the map.")


class Location(BaseModel):
    name: str = Field(
        description="The name of the location extracted from the map area. "
        + "The name should be the name of the point and the index of the point in the extracted text."
    )
    index: int = Field(description="The index of the point in the extracted text.")


class PointsLLM(BaseModel):
    points: List[Location] = Field(
        description="The list of point places extracted from the map area. "
        + "The 'name' key should contain the name of the point and the 'index' key should contain the index of "
        + "the point in the extracted text."
        + "Examples of places that are points: mountains, peaks, trailheads, hills, summits.\n"
        + "Examples of places that are not points: pond, brook, lake, river.\n"
    )


class PopulationCenterLLM(BaseModel):
    population_centers: List[Location] = Field(
        description="The list of recognizeable population centers extracted from the map area. "
        + "The 'name' key should contain the name of the population center and the 'index' key should contain the "
        + "index of the population center in the extracted text."
        + "Examples of population centers: cities, towns, villages, hamlets.\n"
        + "Examples of places that are not population centers: roads, streets, avenues, or other similar features.\n"
    )


class MetadataExtractor(Task):
    # matcher for alphanumeric strings
    ALPHANUMERIC_PATTERN = re.compile(r".*[a-zA-Z].*\d.*|.*\d.*[a-zA-Z].*|.*[a-zA-Z].*")

    # patterns for scale normalization
    SCALE_PATTERN = re.compile(r"[,\. a-zA-z]+")
    SCALE_PREPEND = re.compile(r"\d+:")

    # quadrangle normalization
    QUADRANGLE_PATTERN = re.compile(re.escape("quadrangle"), re.IGNORECASE)

    # max number of tokens allowed by openai api, leaving enough for output
    TOKEN_LIMIT = 3500

    # OCR text filtering control
    MAX_TEXT_FILTER_LENGTH = 600
    MIN_TEXT_FILTER_LENGTH = 100
    TEXT_FILTER_DECREMENT = 100

    TEXT_EXTRACT_TEMPLATE = (
        "The following blocks of text were extracted from a map using an OCR process:\n"
        + "{text_str}"
        + "\n\n"
        + "Extract metadata defined in the output structure from the text.\n"
        + "{format}"
        + "\n"
        + 'If any string value is not present the field should be set to "NULL"\n'
    )

    POINT_PLACE_TEMPLATE = (
        "The following blocks of text were extracted from a map using an OCR process, specified "
        + "as a list with (text, index):\n"
        + "{text_str}"
        + "\n\n"
        + "Extract places that are points from the text.\n"
        + "{format}"
    )

    POPULATION_CENTER_TEMPLATE = (
        "The following blocks of text were extracted from a map using an OCR process, "
        + "specified as a list with (text, index):\n"
        + "{text_str}"
        + "\n\n"
        + " Extract the places that are recognizable metropolitan areas, cities, towns, or villages.\n"
        + "{format}"
    )

    EXAMPLE_JSON_UTM = json.dumps({"utm_zone": "<utm zone>"})

    EXAMPLE_JSON_QUADRANGLES = json.dumps(
        {"quadrangles": ["<quadrangle>", "<quadrangle>"]}
    )

    # threshold for determining map shape - anything above is considered rectangular
    RECTANGULARITY_THRESHOLD = 0.9

    def __init__(
        self,
        id: str,
        model=LLM.GPT_4_TURBO,
        text_key=TEXT_EXTRACTION_OUTPUT_KEY,
        should_run: Optional[Callable] = None,
    ):
        super().__init__(id)

        self._chat_model = ChatOpenAI(
            model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1
        )
        self._model = model
        self._text_key = text_key
        self._should_run = should_run

        logger.info(f"Using model: {self._model.value}")

    def run(self, input: TaskInput) -> TaskResult:
        """Processes a directory of OCR files and writes the metadata to a json file"""
        if self._should_run and not self._should_run(input):
            return self._create_result(input)

        task_result = TaskResult(self._task_id)

        doc_text: DocTextExtraction = input.parse_data(
            TEXT_EXTRACTION_OUTPUT_KEY, DocTextExtraction.model_validate
        )
        if not doc_text:
            return task_result

        # post-processing and follow on prompts
        metadata = self._process_doc_text_extraction(doc_text)
        if metadata:
            # map state names as needed
            for i, p in enumerate(metadata.states):
                if p.lower() in PLACE_EXTENSION_MAP:
                    metadata.states[i] = PLACE_EXTENSION_MAP[p.lower()]

            # normalize scale
            metadata.scale = self._normalize_scale(metadata.scale)

            # normalize quadrangle
            metadata.quadrangles = self._normalize_quadrangle(metadata.quadrangles)

            # # extract places
            metadata.places = self._extract_locations(
                doc_text, self.POINT_PLACE_TEMPLATE
            )

            metadata.population_centres = self._extract_locations(
                doc_text, self.POPULATION_CENTER_TEMPLATE
            )

            # # extract quadrangles from the title and base map info
            # metadata.quadrangles = self._extract_quadrangles(
            #     metadata.title, metadata.base_map
            # )

            # # extract UTM zone if not present in metadata after initial extraction
            # if metadata.utm_zone == "NULL":
            #     metadata.utm_zone = self._extract_utm_zone(metadata)

            # compute map shape from the segmentation output
            segments = input.data[SEGMENTATION_OUTPUT_KEY]
            metadata.map_shape = self._compute_shape(segments)

            # compute map chroma from the image
            metadata.map_chroma = self._compute_chroma(input.image)

            task_result.add_output(
                METADATA_EXTRACTION_OUTPUT_KEY, metadata.model_dump()
            )

        return task_result

    def _process_doc_text_extraction(
        self, doc_text_extraction: DocTextExtraction
    ) -> Optional[MetadataExtraction]:
        """Extracts metadata from a single OCR file"""
        try:
            logger.info(f"Processing '{doc_text_extraction.doc_id}'")

            max_text_length = self.MAX_TEXT_FILTER_LENGTH
            num_tokens = 0

            prompt_str = ""
            input_prompt: Optional[PromptValue] = None
            text = []

            # setup the output structure
            parser = PydanticOutputParser(pydantic_object=MetdataLLM)

            # setup the prompt template
            prompt_template = self._generate_prompt_template(
                parser, self.TEXT_EXTRACT_TEMPLATE
            )

            while max_text_length > self.MIN_TEXT_FILTER_LENGTH:
                # extract text from OCR output using rule-based filtering
                text = self._extract_text(doc_text_extraction, max_text_length)
                input_prompt = prompt_template.format_prompt(text_str="\n".join(text))
                if input_prompt is None:
                    logger.warn(
                        f"Skipping extraction '{doc_text_extraction.doc_id}' - prompt generation failed"
                    )
                    return self._create_empty_extraction(doc_text_extraction.doc_id)

                # if the token count is greater than the limit, reduce the max text length
                # and try again
                num_tokens = self._count_tokens(input_prompt.to_string(), "cl100k_base")
                if num_tokens <= self.TOKEN_LIMIT:
                    break
                max_text_length = max_text_length - self.TEXT_FILTER_DECREMENT
                logger.debug(
                    f"Token count after filtering exceeds limit - reducing max text length to {max_text_length}"
                )

            logger.info(f"Processing {num_tokens} tokens.")

            logger.debug("Prompt string:\n")
            logger.debug(prompt_str)

            # generate the response
            if input_prompt is not None:
                chain = prompt_template | self._chat_model | parser
                response = chain.invoke({"text_str": "\n".join(text)})
                # add placeholders for fields we don't extract
                response_dict = response.dict()
                response_dict["quadrangles"] = []
                response_dict["population_centres"] = []
                response_dict["places"] = []
                response_dict["map_shape"] = "unknown"
                response_dict["map_chroma"] = "unknown"
                response_dict["utm_zone"] = "NULL"
                return MetadataExtraction(
                    map_id=doc_text_extraction.doc_id, **response_dict
                )

            logger.warn(
                f"Skipping extraction '{doc_text_extraction.doc_id}' - input token count {num_tokens} is greater than limit {self.TOKEN_LIMIT}"
            )
            return self._create_empty_extraction(doc_text_extraction.doc_id)

        except Exception as e:
            # print exception stack trace
            logger.error(
                f"Skipping extraction '{doc_text_extraction.doc_id}' - unexpected error during processing",
                exc_info=True,
            )
            return self._create_empty_extraction(doc_text_extraction.doc_id)

    def _extract_locations(
        self, doc_text: DocTextExtraction, template: str
    ) -> List[TextExtraction]:
        text_indices = self._extract_text_with_index(doc_text)

        parser = PydanticOutputParser(pydantic_object=PointsLLM)
        prompt_template = self._generate_prompt_template(parser, template)
        chain = prompt_template | self._chat_model | parser
        response = chain.invoke({"text_str": text_indices})

        return self._map_text_coordinates(response.points, doc_text, False)

    # def _extract_utm_zone(self, metadata: MetadataExtraction) -> str:
    #     """Extracts the UTM zone from the metadata if it is not already present"""
    #     prompt_str = self._to_utm_prompt_str(
    #         metadata.counties,
    #         metadata.quadrangles,
    #         metadata.states,
    #         metadata.places,
    #         metadata.population_centres,
    #     )
    #     utm_zone_resp = self._process_basic_prompt(prompt_str)
    #     if utm_zone_resp == "NULL":
    #         return utm_zone_resp
    #     utm_json = json.loads(utm_zone_resp)
    #     return utm_json["utm_zone"]

    # def _extract_quadrangles(self, title: str, base_map: str) -> List[str]:
    #     """Extracts quadrangles from the title and base map info"""
    #     prompt_str = self._to_quadrangle_prompt_str(title, base_map)
    #     quadrangle_resp = self._process_basic_prompt(prompt_str)
    #     if quadrangle_resp == "NULL":
    #         return []
    #     quadrangle_json = json.loads(quadrangle_resp)
    #     return quadrangle_json["quadrangles"]

    def _extract_text_with_index(
        self, doc_text_extraction: DocTextExtraction
    ) -> List[Tuple[str, int]]:
        # map all text with index
        return [(d.text, i) for i, d in enumerate(doc_text_extraction.extractions)]

    def _text_extractions_to_str(self, extractions: List[Tuple[str, int]]) -> str:
        # want to end up with a list of (text, coordinate) having each entry be a new line
        items = [f"({r[0]}, {i})" for i, r in enumerate(extractions)]
        return "\n".join(items)

    def _map_text_coordinates(
        self,
        places: List[Location],
        extractions: DocTextExtraction,
        replace_text: bool,
    ) -> List[TextExtraction]:
        # want to use the index to filter the extractions
        # TODO: MAY WANT TO CHECK THE TEXT LINES UP JUST IN CASE THE LLM HAD A BIT OF FUN
        filtered = []
        for p in places:
            e = copy.deepcopy(extractions.extractions[p.index])
            if replace_text:
                e.text = p.name
            filtered.append(e)
        return filtered  # type: ignore

    def _count_tokens(self, input_str: str, encoding_name: str) -> int:
        """Counts the number of tokens in a input string using a given encoding"""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(input_str))
        return num_tokens

    def _generate_prompt_template(self, parser, template: str) -> ChatPromptTemplate:
        system_message = "You are using text extracted from geological maps by an OCR process to identify map metadata"
        human_message_template = HumanMessagePromptTemplate.from_template(template)
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=system_message),
                human_message_template,
            ],
            input_variables=["text_str"],
            partial_variables={"format": parser.get_format_instructions()},
        )
        return prompt

    def _to_utm_prompt_str(
        self,
        counties: List[str],
        quadrangles: List[str],
        state: List[str],
        places: List[TextExtraction],
        population_centers: List[TextExtraction],
    ) -> str:
        return (
            "The following information was extracted froma map using an OCR process:\n"
            + f"quadrangles: {','.join(quadrangles)}\n"
            + f"counties: {','.join(counties)}\n"
            + f"places: {','.join([p.text for p in places])}\n"
            + f"population centers: {','.join([p.text for p in population_centers])}\n"
            + f"states: {','.join(state)}\n"
            + " Infer the UTM zone and return it in a JSON structure. If it cannot be inferred, return 'NULL'.\n"
            + " The inferred UTM zone should not include an N or S after the number.\n"
            + " Here is an example of the structure to return: \n"
            + self.EXAMPLE_JSON_UTM
        )

    def _to_quadrangle_prompt_str(self, title: str, base_map_info: str) -> str:
        return (
            "The following information was extracted from a map using an OCR process:\n"
            + f"title: {title}\n"
            + f"base map: {base_map_info}\n"
            + " Identify the quadrangles from the fields and store them in a JSON structure.\n"
            + " Here is an example of the structure to use: \n"
            + self.EXAMPLE_JSON_QUADRANGLES
            + "\n"
        )

    def _extract_text(
        self, doc_text_extraction: DocTextExtraction, max_text_length=800
    ) -> List[str]:
        """Extracts text from OCR output - filters to alphanumeric strings between 4 and 400 characters long
        that contain at least one space"""
        text_dims = []
        for text_entry in doc_text_extraction.extractions:
            text = text_entry.text
            if (
                self.ALPHANUMERIC_PATTERN.match(text)
                and len(text) >= 4
                and len(text) <= max_text_length
                and len(text.split(" ")) > 1
            ):
                text_dims.append(text)
        return text_dims

    def _normalize_scale(self, scale_str: str) -> str:
        """Normalizes the scale string to the format 1:xxxxx"""
        if scale_str != "NULL":
            normalized_scale = re.sub(self.SCALE_PATTERN, "", scale_str)
            if not re.match(self.SCALE_PREPEND, normalized_scale):
                normalized_scale = "1:" + normalized_scale
            return normalized_scale
        return scale_str

    def _normalize_quadrangle(self, quadrangles_str: List[str]) -> List[str]:
        """Normalizes the quadrangle string by removing the word quadrangle"""
        return [
            re.sub(self.QUADRANGLE_PATTERN, "", quad_str).strip()
            for quad_str in quadrangles_str
        ]

    def _compute_shape(self, segments) -> MapShape:
        """
        Computes the shape of the map from the segmentation output using a rectangularity metric

        Args:
            segments: The segmentation output

        Returns:
            MapShape: The shape of the map
        """
        if segments:
            map_segmentation = MapSegmentation.model_validate(segments)
            for segment in map_segmentation.segments:
                if segment.class_label == "map":
                    box_area = (segment.bbox[2] - segment.bbox[0]) * (
                        segment.bbox[3] - segment.bbox[1]
                    )
                    rectangularity = segment.area / box_area
                    if rectangularity > self.RECTANGULARITY_THRESHOLD:
                        map_shape = MapShape.RECTANGULAR
                    else:
                        map_shape = MapShape.IRREGULAR
                    break
        return map_shape

    def _compute_chroma(
        self, input_image: PILImage, max_dim=500, mono_thresh=20, low_thresh=60
    ) -> MapChromaType:
        """
        Computes the chroma of the map image using the LAB color space
        and the centroid of the a and b channels

        Args:
            input_image (PILImage): The map image
            max_dim (int): The maximum dimension for resizing the image

        Returns:
            MapChromaType: The chroma type of the map
        """
        if max_dim > 0:
            # uniformly resize the image so that major axis is max_dim
            image = pil_to_cv_image(input_image)
            h, w, _ = image.shape
            if h > w:
                image = cv2.resize(image, (max_dim, int(h / w * max_dim)))
            else:
                image = cv2.resize(image, (int(w / h * max_dim), max_dim))

        # exract the a and b channels and find the centroid
        cs_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        cs_image = cs_image[:, :, 1:].flatten().reshape(-1, 2)

        # compute the error between the mean and each pixel
        mean_vec = np.sum(cs_image, axis=0) / len(cs_image)
        dist = np.linalg.norm(cs_image - mean_vec, axis=1)

        # square the distance and take the mean
        error = np.mean(dist**2)

        # classify the chroma based on the error
        if error < mono_thresh:
            return MapChromaType.MONO_CHROMA
        elif error < low_thresh:
            return MapChromaType.LOW_CHROMA
        else:
            return MapChromaType.HIGH_CHROMA

    @staticmethod
    def _create_empty_extraction(doc_id: str) -> MetadataExtraction:
        """Creates an empty metadata extraction object"""
        return MetadataExtraction(
            map_id=doc_id,
            title="",
            authors=[],
            year="",
            scale="",
            quadrangles=[],
            datum="",
            projection="",
            coordinate_systems=[],
            utm_zone="",
            base_map="",
            counties=[],
            states=[],
            population_centres=[],
            country="",
            places=[],
            publisher="",
            map_shape=MapShape.UNKNOWN,
            map_chroma=MapChromaType.UNKNOWN,
        )
