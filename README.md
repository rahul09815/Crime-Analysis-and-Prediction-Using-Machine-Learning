# Create Python 3.7 Environment in Anaconda
    
    conda create -n crime_env_37 python=3.7 -y

# Conda Activate Crime Environment
    
    conda activate crime_env_37


# Install Required Packages
    
    pip install Pillow Django djangorestframework django-extensions django-crispy-forms numpy pandas imutils django-cors-headers matplotlib seaborn coreapi pyyaml scikit-learn

 
### Command to Run Server
    
    python manage.py runserver 0.0.0.0:8000


## Visit the server

    http://127.0.0.1:8000/
