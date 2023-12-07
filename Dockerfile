# Use a base image with Python installed
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Specify the command to run your application (replace with your actual command)
CMD ["python", "script.py"]
