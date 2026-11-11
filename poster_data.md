The model was trained on the ciphertext obtained through the use of the aes encryption scheme. The model was trained on two ciphertexts, C1 which was generated through the use of the encryption scheme. Among the 16 bytes of the output, the first byte was taken and that value was XORed with all the combinations input difference in order to train the model and then find the optimal input difference since the paper that we studied only gave the results for 0x80. In this manner, the accuracy and the tnr and the tpr values were calculated and then evaluated across different architectures. 

* Architectures: We trained and compared two distinct neural distinguisher architectures: DBNet (a standard deep CNN baseline) and ND-Gohr (the seminal neural distinguisher architecture proposed by Gohr).  
* Dataset Size: Generated 10^7 ciphertext pairs per difference hypothesis (totaling 2.56 billion pairs). Each class is balanced (50% real cipher pairs, 50% random pairs) and split 80/20 into train and test sets.  
* Ablation: We evaluated the impact of the model architecture (DBNet vs. ND-Gohr) and the decision threshold (`thr_0.5` vs. `thr_calibrated`) to isolate how model complexity and post-hoc calibration affect the resulting distinguisher accuracy.  
* Evaluation Protocol: For every one of the 256 difference models (0x01 to 0xFF), we report metrics under both thresholds. This explicitly exposes degenerate predictors where the model achieves high accuracy solely by class collapse (e.g., TPR ≈ 1.0, TNR ≈ 0.06), ensuring we do not overstate the results.

Based on these values, the best input difference was calculated to be 0x63 without the use of mix columns, and the best difference came out to be 0x34 while using mix columns using the ND Gohr architecture. The impact of hamming weights of the different input differences were also observed but no significant impact was seen across the different values. 

**Motivation**:

-  Finding better architectures for differential linear analysis relative to NDGohr’s resnet  
- Carrying out differential  linear attacks  on AES

**Results**:

Core results From the four experiments:

| Architecture | MC | TPR | TNR | ACC |
| ----- | ----- | ----- | ----- | ----- |
| DBiTNet | ON | 100% | 0% | 50% |
| NDGohr | ON | 100% | 0% | 50% |
| DBiTNet | OFF | 99.38% | 6.64% | 53.01% |
| NDGohr | OFF | 100% | 6.09% | 53.04% |

### **What does this tell us?**

* Extremely high TPR: Both architectures identify almost every positive sample correctly.  
* Extremely low TNR: Both architectures misclassify most negative samples as positive.  
* Low accuracy: Accuracy around 50–53% indicates that the models are performing only marginally better than random guessing on this balanced test set.

In fact, the MC-enabled models predict every sample as positive. This explains their 100% TPR, 0% TNR and exactly 50% accuracy.

–DBiTNet vs NDGohr:

| Property | DBiTNet | NDGohr |
| ----- | ----- | ----- |
| Total parameters | 15,553 | 144,465 |
| Trainable parameters | 15,105 | 143,857 |
| Median inference latency | 7.22 ms | 12.04 ms |
| P95 inference latency | 7.36 ms | 12.26 ms |
| Throughput (samples/s) | 492,606 | 341,136 |
| Accuracy (MC ON) | 50.00% | 50.00% |
| Accuracy (MC OFF) | 53.01% | 53.04% |
| TPR (MC OFF) | 99.38% | 100.00% |
| TNR (MC OFF) | 6.64% | 6.09% |

Observations:

1. Model size: DBiTNet has approximately 9.3 times fewer parameters than NDGohr.  
2. Inference speed: DBiTNet has approximately 44% higher inference throughput.  
3. Classification: NDGohr has marginally higher accuracy, while DBiTNet has marginally higher TNR.  
4. Efficiency: DBiTNet offers substantially better computational efficiency, with nearly identical overall classification accuracy.  
5. Monte Carlo: Both models achieve 50% accuracy when Monte Carlo is enabled, indicating that neither distinguishes the two classes under this configuration.

– results for all 255 input differences (0x01–0xFF). The following table combines these results with the previously analysed MC-disabled NDGohr and DBiTNet

| Metric | NDGohr | DBiTNet |
| ----- | ----- | ----- |
| Total parameters | 144,465 | 15,553 |
| Trainable parameters | 143,857 | 15,105 |
| Average best accuracy | 53.0211% | 51.8385% |
| Best individual accuracy | 53.1546% | 52.3030% |
| Worst individual accuracy | 52.8890% | 51.4320% |
| Input differences evaluated | 255 | 255 |

### **Results for Plaintext Difference 0x63**

The experimental results for the five ciphertext byte pairs (0,1), (2,3), (4,5), (6,7), and (8,9) show consistently poor classification performance for the three-round AES configuration with final-round MixColumns disabled.

All experiments used 100,000 training samples, 10,000 validation samples, 5,000 calibration samples, and 10,000 test samples, with five training epochs.

At the default classification threshold of 0.5, accuracy ranged from 49.25% to 50.40%, with an average of 49.984%. After threshold calibration, accuracy ranged from 49.71% to 50.94%, with an average of 50.172%.

The TPR and TNR values also exhibit substantial variation across byte pairs. Some configurations favour positive predictions, while others favour negative predictions. However, their balanced classification performance remains close to random guessing.

The best default-threshold result was obtained for ciphertext bytes (4,5), achieving 50.40% accuracy. The best calibrated result was obtained for bytes (8,9), achieving 50.94%.

Overall, none of the five byte pairs approaches the paper reference accuracy of 53.09%. The results indicate that the current experimental configuration does not demonstrate effective three-round AES differential classification using these ciphertext byte pairs and the plaintext difference 0x63.

**References:**

* **Bardeh, N. G., & Rijmen, V.** (2021). New key-recovery attack on reduced-round AES. *IACR Transactions on Symmetric Cryptology*, *2021*(3)  
* **Chen, Y., Bao, Z., Shen, Y., & Yu, H.** (2022). A deep learning aided key-recovery framework for large-state block ciphers: Application to round-reduced large-state Speck. *Scientia Sinica Informatica*, *52*(10), 1845–1869. https\://doi.org/10.1360/SSI-2022-0298   
* **Dunkelman, O., & Weizman, A.** (2022). *Another look at differential-linear attacks*. IACR Cryptology ePrint Archive. https\://eprint.iacr.org/\[4\]  
* **Gérault, D., Hambitzer, A., Huppert, M., & Picek, S.** (2025). *Survey: Six years of neural differential cryptanalysis*. IACR Cryptology ePrint Archive. https\://eprint.iacr.org/\[5\]  
* **Zhang, L., Wang, Z., & Lu, J.** (2024). Differential-neural cryptanalysis on AES. *IEICE Transactions on Information and Systems*, *E107-D*(10), 1372–1375. 

