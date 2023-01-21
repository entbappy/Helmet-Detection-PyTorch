import sys
from helmet.components.data_ingestion import DataIngestion
from helmet.components.data_transformation import DataTransformation
from helmet.components.model_trainer import ModelTrainer
from helmet.components.model_evaluation import ModelEvaluation
from helmet.components.model_pusher import ModelPusher
from helmet.configuration.s3_operations import S3Operation
from helmet.entity.config_entity import DataIngestionConfig,DataTransformationConfig, ModelTrainerConfig,ModelEvaluationConfig, ModelPusherConfig
from helmet.entity.artifacts_entity import DataIngestionArtifacts, DataTransformationArtifacts, ModelTrainerArtifacts, ModelEvaluationArtifacts,ModelPusherArtifacts 
from helmet.logger import logging
from helmet.exception import HelmetException




class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.s3_operations = S3Operation()


    
    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from S3 bucket")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config, s3_operations= S3Operation()
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train, test and valid from s3")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e


    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifacts,) -> DataTransformationArtifacts:
        logging.info(
            "Entered the start_data_transformation method of TrainPipeline class"
        )
        try:
            data_transformation = DataTransformation(
                
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )
            return data_transformation_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e

    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        logging.info(
            "Entered the start_model_trainer method of TrainPipeline class"
        )
        try:
            model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifact,
                                        model_trainer_config=self.model_trainer_config
                                        )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifact

        except Exception as e:
            raise HelmetException(e, sys)

    
    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifacts, data_transformation_artifact: DataTransformationArtifacts) -> ModelEvaluationArtifacts:
        logging.info("Entered the start_model_evaluation method of TrainPipeline class")
        try:
            model_evaluation = ModelEvaluation(data_transformation_artifacts = data_transformation_artifact,
                                                model_evaluation_config=self.model_evaluation_config,
                                                model_trainer_artifacts=model_trainer_artifact)

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Exited the start_model_evaluation method of TrainPipeline class")
            return model_evaluation_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e


    
    def start_model_pusher(self,s3: S3Operation,) -> ModelPusherArtifacts:
        logging.info("Entered the start_model_pusher method of TrainPipeline class")
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                s3=s3,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Initiated the model pusher")
            logging.info("Exited the start_model_pusher method of TrainPipeline class")
            return model_pusher_artifact

        except Exception as e:
            raise HelmetException(e, sys) from e


    
    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            model_evaluation_artifact = self.start_model_evaluation(model_trainer_artifact=model_trainer_artifact,
                                                                    data_transformation_artifact=data_transformation_artifact
            )
            if not model_evaluation_artifact.is_model_accepted:
                 raise Exception("Trained model is not better than the best model")
            
            model_pusher_artifact = self.start_model_pusher(s3=self.s3_operations)
            
            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise HelmetException(e, sys) from e