FROM python:3.8

# Install PyTorch
RUN pip install numpy pandas torch

RUN pip install transformers scikit-learn


# Copy the notebook into the container
ADD fetch_takehome_exam.py .


CMD ["python", "./fetch_takehome_exam.py"]