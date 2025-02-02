# GEO 225E - Probability and Statistics Assignment

## 📌 Overview
This repository contains my assignment for the **GEO 225E - Probability and Statistics** course at **Istanbul Technical University**. The project involves statistical analysis of **discrete and continuous random variables**, focusing on **Poisson and Normal distributions**, along with **Bivariate Normal Distribution**.

## 📂 Contents
- **Poisson Distribution**
  - Generating random datasets with Poisson distribution
  - Computing descriptive statistics (mean, median, range, standard deviation, quartiles)
  - Visualizing histograms and frequency tables
  
- **Normal Distribution**
  - Creating random datasets with specific mean and variance
  - Computing descriptive statistics
  - Plotting boxplots for data visualization
  
- **Bivariate Normal Distribution**
  - Generating random samples from a bivariate normal distribution
  - Mathematical derivation of the CDF
  - Expected values calculation
  
## 📜 Files Included
- `010220507STA.docx` - The assignment document explaining the methodology
- `codes.docx` - Additional documentation for the implemented codes
- `HW1.py` - Python script containing all statistical computations and visualizations

## 🛠 Technologies Used
- **Python**
- **NumPy** - For numerical computations
- **Matplotlib** - For data visualization
- **Spyder** (IDE used for coding and debugging)

## 📊 Implementation Details
- Poisson distribution samples generated using:
  ```python
  data_poisson_30 = np.random.poisson(mu, 30)
  ```
- Normal distribution samples generated using:
  ```python
  data_normal_50 = np.random.normal(mu, np.sqrt(float(sigma2)), 50)
  ```
- Descriptive statistics calculated using:
  ```python
  def descriptive_stats(data):
      return {
          "Mean": np.mean(data),
          "Median": np.median(data),
          "Range": np.ptp(data),
          "Standard Deviation": np.std(data),
          "Quartiles": list(np.percentile(data, [25, 50, 75]))
      }
  ```

## 📌 Conclusion
This assignment demonstrates the application of probability theory in **Geomatics Engineering**, leveraging **Python for statistical analysis and visualization**. The provided scripts allow for a structured approach to handling probability distributions and their real-world applications.
