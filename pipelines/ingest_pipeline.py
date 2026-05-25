from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":

    ingestion = DataIngestion()

    ingestion.initiate_data_ingestion()

    print("Data ingestion completed")