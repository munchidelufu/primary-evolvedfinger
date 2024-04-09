# Exploiting Multi-domain Deep Model Intellectual Property Protection From a Training-Free Fingerprinting Strategy
The high cost of developing high-performance deep models highlights their value as intellectual property for creators. However, it is important to consider the potential risks of theft. Although various techniques have been developed to protect the intellectual property of deep models, there is still room for improvement in terms of efficiency, comprehensiveness, and generalization. To address the aforementioned limitations, we would like to suggest two efficient algorithms for generating fingerprinting samples, where the first one possesses the advantage of efficiency, while the second one is better in terms of robustness. The first algorithm takes a comprehensive approach to modelling the fingerprint of the deep model. The generated samples are distributed within the stable region and near the decision boundary of the model, while taking into account both the duality and conviction factors. Then, a heuristic sample perturbation algorithm is introduced, which generates a fingerprint with solid stability and generalization across multiple domains. The effectiveness and efficiency of our proposed Primary Fingerprint and Evolved Fingerprint have been demonstrated through simulations of six intellectual property removal attacks and three domain adaptive intellectual property detection attacks in computer vision, natural language processing, and brain-computer interface.

# Preparation
```
pip install -r requirements.txt
```
# How to run the code
Prepare data
Download the original data and put it in the ./data directory.

## Prepare source model
```
python ./cv/source.py
python ./nlp/source.py
python ./bci/source.py
```
## Attack the source model
```
python ./cv/fine_tune.py
python ./nlp/fine_tune.py

detect_erase_attack.py is the IP Detection and erasure attacks.
...
```
## Primary-Fingerprint
```
python .primary.py
```
## Evolved-Fingerprint
```
python.evolved.py
```
## ModelFingerprintMatch
```
python .model_finger_match.py
```
## note
CV code can find at SAC.
