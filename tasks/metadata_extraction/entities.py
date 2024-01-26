from tasks.text_extraction.entities import TextExtraction

from typing import List, Tuple
from pydantic import BaseModel, ConfigDict


METADATA_EXTRACTION_OUTPUT_KEY = "metadata_extraction_output"
GEOCODED_PLACES_OUTPUT_KEY = "geocoded_places_output"


class MetadataExtraction(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    map_id: str
    title: str
    authors: List[str]
    year: str  # should be an int, but there's a chance somethign else is (incorrectly) extracted
    scale: str  # of the format 1:24000
    quadrangles: List[str]
    datum: str
    vertical_datum: str
    projection: str
    coordinate_systems: List[str]
    base_map: str
    counties: List[str]
    states: List[str]
    country: str
    places: List[
        TextExtraction
    ]  # a list of places, each place having a name and coordinates


class GeocodedCoordinate(BaseModel):
    pixel_x: float
    pixel_y: float
    geo_x: float
    geo_y: float


class GeocodedPlace(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    place_name: str
    place_location_restriction: str  # restrict the search space when geocoding to the country or the state to reduce noise
    place_type: str  # either bound for places that are not on the map but restrict the coordinate space, or point / line / polygon
    coordinates: List[
        List[GeocodedCoordinate]
    ]  # bounds will not have the pixel coordinates defined, with each geocoding option a separate list of coordinates


class DocGeocodedPlaces(BaseModel):
    map_id: str
    places: List[GeocodedPlace] = []
