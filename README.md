# Intellectual Property Protection for Deep Models: Pioneering Cross-Domain Fingerprinting Solutions
The high cost of developing high-performance deep models highlights their value as intellectual property for creators. However, it is important to consider the potential risks of theft. Although various techniques have been developed to protect the intellectual property of deep models, there is still room for improvement in terms of efficiency, comprehensiveness, and generalization. Compared with the intrusiveness of watermarking methods, fingerprinting methods do not affect the training process of the source model. Consequently, this paper proposes a fingerprinting method to address the paucity of attempts in fingerprinting methods for model protection. Our method consists of two efficient algorithms for generating fingerprinting samples, where the first one possesses the advantage of efficiency, while the second one is better in terms of robustness. The first algorithm takes a comprehensive approach to modeling the fingerprint of the deep model. The generated samples are distributed within the stable region and near the decision boundary of the model, taking into account both the duality and the conviction factors. Then, a heuristic sample perturbation algorithm is introduced, which generates a fingerprint with solid stability and generalization across multiple domains. The two algorithms proposed in this paper have been shown to be capable of withstanding attacks on intellectual property removal, detection, and evasion. They also show some advantages in terms of efficiency. In addition, the proposed method is the first to apply fingerprinting techniques in a cross-domain context.

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
python .evolved.py
```
## ModelFingerprintMatch
```
python .model_finger_match.py
```
## note
CV code can find at SAC, we reuse the code.
