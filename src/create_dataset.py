import hydra
from typing import NoReturn
from src.enities.feature_extraction_params import FeatureExtractionParams
from src.data.make_dataset import extract_features
from src.descriptions.descriptions import ORB, SIFT, DescriptionsTypes


@hydra.main(version_base=None, config_path='../configs', config_name='feature_extraction_config')
def create_dataset(params: FeatureExtractionParams) -> NoReturn:
    if params.type_descriptor == DescriptionsTypes.ORB:
        descriptor = ORB()
    elif params.type_descriptor == DescriptionsTypes.SIFT:
        descriptor = SIFT()
    else:
        raise NotImplementedError()

    extract_features(descriptor, params)



if __name__ == '__main__':
    create_dataset()
