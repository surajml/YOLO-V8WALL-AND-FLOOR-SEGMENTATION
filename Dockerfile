FROM python:3.11.1
# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy your application code to the image
COPY . /app

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirement.txt

# Expose the necessary ports (if required)
EXPOSE 5000

# Run your application
CMD python app.py

