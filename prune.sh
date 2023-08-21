#!/bin/bash

function file {
  case $1 in
    1)
      echo "mobilenetv2.pt"
      ;;
    2)
      echo "mobilenetv3_small.pt"
      ;;
    3)
      echo "mobilenetv3_large.pt"
      ;;
    4)
      echo "shufflenetv2.pt"
      ;;
    5)
      echo "squeezenet1_1.pt"
      ;;
    6)
      echo "efficientnet_b0.pt"
      ;;
    7)
      echo "efficientnet_b1.pt"
      ;;
    8)
      echo "resnet18.pt"
      ;;
    9)
      echo "resnet34.pt"
      ;;
    10)
      echo "resnet50.pt"
      ;;
    11)
      echo "vgg11.pt"
      ;;
    12)
      echo "vgg16.pt"
      ;;
    13)
      echo "vgg19.pt"
      ;;
    *)
      echo "Invalid model number." >&2
      exit 1
      ;;
  esac
}


for i in {1..13}; do
    model=$(file $i)  # Ottieni il nome del file del modello corrispondente
    echo -e "\nottimizzo il modello $i: $model\n"
    python3 ottimizzamodello.py $i
done
