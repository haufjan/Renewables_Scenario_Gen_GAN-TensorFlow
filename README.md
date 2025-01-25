# Renewables_Scenario_Gen_GAN-TensorFlow
Unofficial implementation of Renewables Scenario Generation GAN (Chen et al., IEEE 2018) in TensorFlow 2.

Y. Chen, Y. Wang, D. Kirschen and B. Zhang, "Model-Free Renewable Scenario Generation Using Generative Adversarial Networks," in IEEE Transactions on Power Systems, vol. 33, no. 3, pp. 3265-3275, May 2018, doi: 10.1109/TPWRS.2018.2794541

Original Codebase: https://github.com/chennnnnyize-zz/Renewables_Scenario_Gen_GAN

### Data Set Reference
* Solar Data: https://www.nrel.gov/grid/solar-power-data.html
* Wind Data: https://www.nrel.gov/grid/wind-integration-data.html

### Results
The recreated experiment demonstrates the GAN's capability to resemble the data distribution and leverage the provided labels for conditioned generation of certain sceanrios after short training trials [Notebook](./renewables_scenario_gen_gan.ipynb). Obviously, the training can be extended and parameters adapted to improve the GAN's fidelity.

The plots present synthesized solar power generation scenarios arranged by ordinal label values.

![generated_data](../assets/generated_data.png)
