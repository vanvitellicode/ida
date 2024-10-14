# IDA
Information Disorder Awarenes Micro Service

# Instructions for using the service

1. **Clone the project repository:**

   Download the repository from the remote source.

2. **Navigate to the `UseCasesCode` directory:**

   ```
   cd UseCasesCode
   ```

3. **Start the service using Docker Compose**

   Depending on whether you want to tow the model or test it on a dataset write the following commands:
   
   For train:

   ```
   docker-compose up train_model
   ```
   For test:
   ```
   docker-compose up use_model
   ```

   
