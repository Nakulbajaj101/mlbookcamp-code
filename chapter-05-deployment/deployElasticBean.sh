#!/bin/bash

# Create subnets so service can be scaled as load balancer need couple of subnets

aws ec2 create-default-subnet --availability-zone us-east-1a
aws ec2 create-default-subnet --availability-zone us-east-1b
aws ec2 create-default-subnet --availability-zone us-east-1c
aws ec2 create-default-subnet --availability-zone us-east-1d
aws ec2 create-default-subnet --availability-zone us-east-1e
aws ec2 create-default-subnet --availability-zone us-east-1f

# Install pipenv

pip install pipenv

# Install pipenv dependencies
pipenv install    

# Activate local pipenv environment
pipenv shell

# Initialise the churn service application 
eb init -p docker -r us-east-1 churn-service    

# Create churn service
eb create churn-service-env
