
import os
import sys
import torch
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from helmet.constants import *
from helmet.logger import logging
from helmet.exception import HelmetException
from helmet.utils.main_utils import load_object
from helmet.entity.config_entity import ModelEvaluationConfig
from helmet.configuration.s3_operations import S3Operation
from helmet.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts


class ModelEvaluation:

    def __init__(self, model_evaluation_config:ModelEvaluationConfig,
                data_transformation_artifacts:DataTransformationArtifacts,
                model_trainer_artifacts:ModelTrainerArtifacts):

        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts
        self.s3 = S3Operation()
        self.bucket_name = BUCKET_NAME


    
    @staticmethod
    def collate_fn(batch):
        """
        This is our collating function for the train dataloader,
        it allows us to create batches of data that can be easily pass into the model
        """
        try:
            return tuple(zip(*batch))
        except Exception as e:
            raise HelmetException(e, sys) from e

    
    def get_model_from_s3(self) -> str:
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            # Loading the best model from s3 bucket
            predict_model_path = self.model_evaluation_config.BEST_MODEL_PATH
            best_model_path = self.s3.read_data_from_s3(TRAINED_MODEL_NAME, self.bucket_name, predict_model_path)
            logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise HelmetException(e, sys) from e


    
    def evaluate(self, model, dataloader, device):
        try:
            model.to(device)
            all_losses = []
            all_losses_dict = []

            for images, targets in tqdm(dataloader):
                images = list(image.to(device) for image in images)
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)  # the model computes the loss automatically if we pass in targets
                losses = sum(loss for loss in loss_dict.values())
                loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
                loss_value = losses.item()

                all_losses.append(loss_value)
                all_losses_dict.append(loss_dict_append)

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")  # train if loss becomes infinity
                    print(loss_dict)
                    sys.exit(1)

                losses.backward()

            all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing

            print("loss: {:.6f},loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
                    np.mean(all_losses),
                    all_losses_dict['loss_classifier'].mean(),
                    all_losses_dict['loss_box_reg'].mean(),
                    all_losses_dict['loss_rpn_box_reg'].mean(),
                    all_losses_dict['loss_objectness'].mean()
                ))
            return all_losses_dict, np.mean(all_losses)

        except Exception as e:
            raise HelmetException(e, sys) from e


    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
                Method Name :   initiate_model_evaluation
                Description :   This function is used to initiate all steps of the model evaluation

                Output      :   Returns model evaluation artifact
                On Failure  :   Write an exception log and then raise an exception
        """

        try:
            trained_model = torch.load(self.model_trainer_artifacts.trained_model_path)

            test_dataset = load_object(self.data_transformation_artifacts.transformed_test_object)

            test_loader = DataLoader(test_dataset,
                                      batch_size=self.model_evaluation_config.BATCH,
                                      shuffle=self.model_evaluation_config.SHUFFLE,
                                      num_workers=self.model_evaluation_config.NUM_WORKERS,
                                      collate_fn=self.collate_fn
                                      )

            logging.info("loaded saved model")

            trained_model = trained_model.to(DEVICE)

            all_losses_dict, all_losses = self.evaluate(trained_model, test_loader, device=DEVICE)
            os.makedirs(self.model_evaluation_config.EVALUATED_MODEL_DIR, exist_ok=True)
            all_losses_dict.to_csv(self.model_evaluation_config.EVALUATED_LOSS_CSV_PATH, index=False)

            s3_model = self.get_model_from_s3()
            s3_model = torch.load(s3_model, map_location=torch.device(DEVICE))

            s3_all_losses_dict, s3_all_losses = self.evaluate(s3_model,test_loader, device=DEVICE)

            if s3_all_losses > all_losses:
                # 0.03 > 0.02
                is_model_accepted = True

                model_evaluation_artifact = ModelEvaluationArtifacts(
                    is_model_accepted=is_model_accepted,
                    all_losses=all_losses)

            else:
                is_model_accepted = False

                model_evaluation_artifact = ModelEvaluationArtifacts(
                    is_model_accepted=is_model_accepted,
                    all_losses=s3_all_losses)

            logging.info("Exited the initiate_model_evaluation method of Model Evaluation class")
            return model_evaluation_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e
