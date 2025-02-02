# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:15:40 2025

@author: ŞENER
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define PDF of X
def pdf_x(x):
    if 0 < x <= 2:
        return (5 / 32) * x**4
    else:
        return 0

# Compute CDF of X using numerical integration
def cdf_x(x):
    return quad(pdf_x, 0, x)[0]

# Generate X values for plotting
x_values = np.linspace(0, 2, 100)

# Compute PDF and CDF values for X
pdf_values = [pdf_x(x) for x in x_values]
cdf_values = [cdf_x(x) for x in x_values]

# Mean value of X (mu)
mu = 2

# Plot the CDF of X
plt.figure(figsize=(8, 6))
plt.plot(x_values, cdf_values, label='CDF of X', color='orange')
plt.axvline(x=mu, color='red', linestyle='--', label='Mean of X ($\mu$)')
plt.title('CDF of X with Mean Value')
plt.xlabel('x')
plt.ylabel('F_X(x)')
plt.legend()
plt.grid()
plt.show()
# Define transformation Y = X^2 + mu and compute CDF for Y
def pdf_y(y):
    # Inverse relationship to find corresponding x for given y
    if y >= mu and y <= 4 + mu:
        x = np.sqrt(y - mu)
        return pdf_x(x) / (2 * np.sqrt(y - mu))
    else:
        return 0

# Compute CDF of Y using numerical integration
def cdf_y(y):
    return quad(pdf_y, mu, y)[0]

# Generate Y values for plotting
y_values = np.linspace(mu, 4 + mu, 100)

# Compute CDF values for Y
cdf_y_values = [cdf_y(y) for y in y_values]

# Mean value of Y
mean_y = np.mean(y_values)

# Plot the CDF of Y
plt.figure(figsize=(8, 6))
plt.plot(y_values, cdf_y_values, label='CDF of Y', color='green')
plt.axvline(x=mean_y, color='blue', linestyle='--', label='Mean of Y ($\mu_Y$)')
plt.title('CDF of Y with Mean Value')
plt.xlabel('y')
plt.ylabel('F_Y(y)')
plt.legend()
plt.grid()
plt.show()
# Step: Create bivariate normal distribution
mean = [0, 1]  # Mean vector
cov = [[1, 0.2], [0.2, 2]]  # Covariance matrix (assuming t=5th digit of ID is 5)

# Generate 200 samples from the bivariate normal distribution
samples = np.random.multivariate_normal(mean, cov, 200)

# Extract X and Y components
x_samples = samples[:, 0]
y_samples = samples[:, 1]

# Plot the scatter plot of the samples
plt.figure(figsize=(8, 6))
plt.scatter(x_samples, y_samples, alpha=0.7, label='Samples')
plt.title('Scatter Plot of Bivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
# Plot the scatter plot of the samples
plt.figure(figsize=(8, 6))
plt.scatter(x_samples, y_samples, alpha=0.7, label='Samples')
plt.title('Scatter Plot of Bivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

# Create a 2D grid for the PDF
x_grid = np.linspace(min(x_samples), max(x_samples), 100)
y_grid = np.linspace(min(y_samples), max(y_samples), 100)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# Calculate the PDF of the bivariate normal distribution
from scipy.stats import multivariate_normal
pdf_values = multivariate_normal(mean, cov).pdf(np.dstack((x_mesh, y_mesh)))

# Plot the PDF
plt.figure(figsize=(8, 6))
plt.contourf(x_mesh, y_mesh, pdf_values, cmap='viridis')
plt.colorbar(label='Density')
plt.title('PDF of Bivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Plot the PDF of the first variable (X)
plt.figure(figsize=(8, 6))
plt.hist(x_samples, bins=30, density=True, color="blue", alpha=0.7, label='PDF of X')
plt.title('PDF of First Variable (X)')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# Plot the PDF of the second variable (Y)
plt.figure(figsize=(8, 6))
plt.hist(y_samples, bins=30, density=True, color="green", alpha=0.7, label='PDF of Y')
plt.title('PDF of Second Variable (Y)')
plt.xlabel('Y')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# Calculate the correlation between X and Y
correlation = np.corrcoef(x_samples, y_samples)[0, 1]
print(f"Correlation between X and Y: {correlation:.2f}")

# Scatter plot with correlation value
plt.figure(figsize=(8, 6))
plt.scatter(x_samples, y_samples, alpha=0.7, label=f'Correlation: {correlation:.2f}')
plt.title('Scatter Plot of X and Y with Correlation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

correlation = np.corrcoef(x_samples, y_samples)[0, 1]
print(f"Correlation between X and Y: {correlation:.2f}")
# Define Z as Z = X + Y
z_samples = x_samples + y_samples

# Calculate E[Z] (Expected Value of Z)
e_z = np.mean(z_samples)

# Calculate VAR[Z] (Variance of Z)
var_z = np.var(z_samples)

print(f"E[Z] (Expected Value of Z): {e_z:.2f}")
print(f"VAR[Z] (Variance of Z): {var_z:.2f}")
# Calculate the correlation and covariance between X and Z
correlation_xz = np.corrcoef(x_samples, z_samples)[0, 1]
covariance_xz = np.cov(x_samples, z_samples)[0, 1]

print(f"Correlation between X and Z: {correlation_xz:.2f}")
print(f"Covariance between X and Z: {covariance_xz:.2f}")

# Scatter plot of X and Z
plt.figure(figsize=(8, 6))
plt.scatter(x_samples, z_samples, alpha=0.7, label=f'Corr: {correlation_xz:.2f}, Cov: {covariance_xz:.2f}')
plt.title('Scatter Plot of X and Z with Correlation and Covariance')
plt.xlabel('X')
plt.ylabel('Z')
plt.legend()
plt.grid()
plt.show()
correlation_xy = np.corrcoef(x_samples, y_samples)[0, 1]
