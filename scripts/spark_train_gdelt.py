
import sys
import logging
from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegressionTrainer:
    """Train regression models for stock price prediction using PySpark"""
    
    def __init__(self, bucket_name, project_id='gdelt-stock-sentiment-analysis'):
        """
        Initialize PySpark regression trainer
        
        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.gcs_path = f"gs://{bucket_name}"
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("GDELT-Regression-Training") \
            .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.5") \
            .getOrCreate()
        
        self.spark.sparkContext._jsc.hadoopConfiguration().set(
            "fs.gs.project.id", project_id
        )
        
        logger.info(f"Spark session initialized for project {project_id}")
    
    def run(self, file_path='gdelt_raw/joined_data.csv', output_path=None):
        """Load data and train regression models for stock price prediction"""
        logger.info("Starting GDELT PySpark regression model training")
        
        try:
            # Load data from GCS
            logger.info(f"Loading data from {self.gcs_path}/{file_path}")
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(f"{self.gcs_path}/{file_path}")
            
            logger.info(f"Loaded {df.count()} rows")
            logger.info(f"Columns: {df.columns}")
            df.printSchema()
            
            # Show sample data
            logger.info("Sample data:")
            df.show(5, truncate=False)
            
            # Define feature and label columns
            # feature_cols = ['company', 'ticker', 'daily_exposure_count', 'daily_avg_tone']
            # label_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [
                "Open", "High", "Low", "prev_close", "Volume",
                "daily_return_pct", "day_of_week",
                "daily_exposure_count", "daily_avg_tone"
            ]
            label_col = "next_day_close"
            
            
            # Verify all columns exist
            missing_features = [col_name for col_name in feature_cols if col_name not in df.columns]
            missing_labels = [label_col] if label_col not in df.columns else []
            
            if missing_features or missing_labels:
                logger.error(f"Missing features: {missing_features}, Missing labels: {missing_labels}")
                raise ValueError(f"Missing required columns")
            
            logger.info(f"Using feature columns: {feature_cols}")
            logger.info(f"Using label columns: {label_col}")
            
            # Drop rows with null values in feature and label columns
            df_clean = df.dropna(subset=feature_cols + [label_col])
            logger.info(f"After removing nulls: {df_clean.count()} rows")
            
            # Handle categorical features (company, ticker)
            logger.info("Encoding categorical features")
            
            # Index categorical columns
            company_indexer = StringIndexer(inputCol='company', outputCol='company_indexed')
            ticker_indexer = StringIndexer(inputCol='ticker', outputCol='ticker_indexed')
            
            df_indexed = company_indexer.fit(df_clean).transform(df_clean)
            df_indexed = ticker_indexer.fit(df_indexed).transform(df_indexed)
            
            # One-hot encode categorical features
            company_encoder = OneHotEncoder(inputCol='company_indexed', outputCol='company_encoded')
            ticker_encoder = OneHotEncoder(inputCol='ticker_indexed', outputCol='ticker_encoded')
            
            df_encoded = company_encoder.fit(df_indexed).transform(df_indexed)
            df_encoded = ticker_encoder.fit(df_encoded).transform(df_encoded)
            
            # Select numeric features and encoded categorical features
            numeric_features = [
                "Open", "High", "Low", "prev_close", "Volume",
                "daily_return_pct", "day_of_week",
                "daily_exposure_count", "daily_avg_tone"
            ]
            all_features = numeric_features + ['company_encoded', 'ticker_encoded']
            
            # Assemble features
            logger.info("Assembling feature vectors")
            assembler = VectorAssembler(
                inputCols=all_features,
                outputCol='raw_features'
            )
            df_assembled = assembler.transform(df_encoded)
            
            # Scale features
            logger.info("Scaling features")
            scaler = StandardScaler(
                inputCol='raw_features',
                outputCol='features',
                withMean=True,
                withStd=True
            )
            scaler_model = scaler.fit(df_assembled)
            df_scaled = scaler_model.transform(df_assembled)
            
            # Split data into train and test (80/20)
            logger.info("Splitting data into train/test sets (80/20)")
            train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)
            logger.info(f"Train set: {train_df.count()} rows")
            logger.info(f"Test set: {test_df.count()} rows")
            
            # Train regression models for each label
            models = {}
            results = {}
            
            logger.info("TRAINING REGRESSION MODELS")
            try:
                logger.info("Training Linear Regression")
                lr = LinearRegression(labelCol=label_col, featuresCol='features', maxIter=100, regParam=0.01)
                lr_model = lr.fit(train_df)
                models['lr'] = lr_model
    
                predictions = lr_model.transform(test_df)
                evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='rmse')
                evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='r2')
                rmse = evaluator_rmse.evaluate(predictions)
                r2 = evaluator_r2.evaluate(predictions)
    
                results['lr'] = {'rmse': rmse, 'r2': r2}
                logger.info(f"✓ Linear Regression - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            except Exception as e:
                logger.error(f"✗ Error training Linear Regression: {e}")
 
            # GBT Regressor
            try:
                logger.info("Training GBT Regressor")
                gbt = GBTRegressor(labelCol=label_col, featuresCol='features', maxIter=50, maxDepth=5, seed=42)
                gbt_model = gbt.fit(train_df)
                models['gbt'] = gbt_model
    
                predictions = gbt_model.transform(test_df)
                evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='rmse')
                evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='r2')
                rmse = evaluator_rmse.evaluate(predictions)
                r2 = evaluator_r2.evaluate(predictions)
    
                results['gbt'] = {'rmse': rmse, 'r2': r2}
                logger.info(f"✓ GBT Regressor - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            except Exception as e:
                logger.error(f"✗ Error training GBT: {e}")
 
            # Random Forest
            try:
                logger.info("Training Random Forest Regressor")
                rf = RandomForestRegressor(labelCol=label_col, featuresCol='features', numTrees=50, maxDepth=10, seed=42)
                rf_model = rf.fit(train_df)
                models['rf'] = rf_model
    
                predictions = rf_model.transform(test_df)
                evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='rmse')
                evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol='prediction', metricName='r2')
                rmse = evaluator_rmse.evaluate(predictions)
                r2 = evaluator_r2.evaluate(predictions)
    
                results['rf'] = {'rmse': rmse, 'r2': r2}
                logger.info(f"✓ Random Forest - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            except Exception as e:
                logger.error(f"✗ Error training Random Forest: {e}")
            # Export predictions for analysis
            logger.info("EXPORTING PREDICTIONS FOR ANALYSIS")

            
            # Use Random Forest model for next_day_close predictions
            best_model = models.get('rf')
            if best_model:
                predictions = best_model.transform(test_df)
                
                # Select required columns
                predictions_export = predictions.select(
                    'event_date',
                    col(label_col).alias('actual'),
                    col('prediction').alias('predicted'),
                    'prev_close',
                    'daily_avg_tone',
                    'company',
                    'ticker'
                ).toPandas()
                
                # Save to GCS
                storage_client = storage.Client(project=self.project_id)
                bucket = storage_client.bucket(self.bucket_name)
                
                # Save locally first
                local_path = "/tmp/predictions.csv"
                predictions_export.to_csv(local_path, index=False)
                
                # Upload to GCS
                if output_path is None:
                    blob_path = "predictions/predictions.csv"
                else:
                    blob_path = f"{output_path}/predictions.csv"
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                
                logger.info(f"✓ Predictions saved to gs://{self.bucket_name}/{blob_path}")
                logger.info(f"  Rows: {len(predictions_export)}, Columns: {list(predictions_export.columns)}")


            # Save models
            if models:
                logger.info("\n" + "="*60)
                logger.info("SAVING MODELS")
                logger.info("="*60)
                if output_path is None:
                    output_path = f"{self.gcs_path}/regression_models"
                else:
                    output_path = f"{self.gcs_path}/{output_path}"
                
                for model_name, model in models.items():
                    try:
                        model_path = f"{output_path}/{model_name}"
                        model.save(model_path)
                        logger.info(f"✓ Saved {model_name} to {model_path}")
                    except Exception as e:
                        logger.error(f"Error saving {model_name}: {e}")
            
            logger.info("\n" + "="*60)
            logger.info("✓ Regression model training completed successfully")
            logger.info("="*60)
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            traceback.print_exc()
            raise
        finally:
            # Stop Spark session
            self.spark.stop()
            logger.info("Spark session stopped")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        logger.error("Usage: python pyspark_ml_train.py <bucket_name> [file_path] [output_path]")
        sys.exit(1)
    
    bucket_name = sys.argv[1]
    file_path = sys.argv[2] if len(sys.argv) > 2 else 'gdelt_raw/joined_data.csv'
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    trainer = RegressionTrainer(
        bucket_name=bucket_name,
        project_id='gdelt-stock-sentiment-analysis'
    )
    
    trainer.run(file_path=file_path, output_path=output_path)


if __name__ == '__main__':
    main()
