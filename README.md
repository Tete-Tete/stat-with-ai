# Go Regression Analysis Tool

This repository contains a Go program that performs statistical analysis on Anscombe's quartet. The Anscombe Quartet is a set of four datasets that have identical statistical properties but vary significantly when visualized. This project demonstrates the importance of visualizing data before making assumptions based solely on statistical analysis. This project mainly created by the Chatgpt.

## Overview
The main goal of this project is to provide a simple and effective tool for performing regression analysis. It aims to help users explore relationships between variables using linear regression techniques. The Anscombe's Quartet consists of four datasets, each having nearly identical statistical properties but appearing very different when graphed. This project helps highlight the importance of visualizing data.

## Project Structure
- **main.go**: Contains the main Go code for statistical analysis and linear regression calculations on the Anscombe datasets.
- **README.md**: Provides an overview of the project, including how to use and test the program.

## Requirements
- Go programming language installed (version 1.16 or higher recommended).
- The required package, `github.com/montanaflynn/stats` and `github.com/stretchr/testify/assert`, should be installed using:
  ```sh
  go get github.com/montanaflynn/stats
  go get github.com/stretchr/testify/assert
  ```

## Installation From Git and Setup
### Step 1: Clone the Repository
Clone this repository to your local machine:
```sh
git clone <https://github.com/Tete-Tete/stat-with-ai.git>
```

### Step 2: Running the Program
To run the analysis on the Anscombe Quartet datasets, execute the following command:
```sh
go run main.go
```

## Running Tests
The repository includes unit tests to validate the linear regression calculations for each dataset. To run the tests, execute the following command:
```sh
go test
```
These tests use the testify/assert package to ensure the computed slope, intercept, and R-squared values are within an acceptable margin of error for each dataset

## Dataset
The Anscombe Quartet datasets are defined in the code as follows:

- **Dataset 1**: `x1` and `y1`
- **Dataset 2**: `x2` and `y2`
- **Dataset 3**: `x3` and `y3`
- **Dataset 4**: `x4` and `y4` (includes an outlier to highlight the difference visually)

## Training Materials
- **Go Documentation**: https://golang.org/doc/ - Useful for understanding Go syntax, standard library functions, and best practices.
- **GitHub Go Packages**: https://pkg.go.dev/ - Used to explore documentation for third-party libraries, including montanaflynn/stats and testify/assert.
- **YouTube Video on Go Testing**:Unit Testing in Golang - Provided insights on how to effectively write unit tests in Go. https://www.youtube.com/watch?v=U-eO9_lNi7w&ab_channel=hatchpad
- **Learn Go with Tests (Book by Chris James)**:GitHub Repository - Helped with understanding test-driven development in Go. https://github.com/quii/learn-go-with-tests

## Sharing Go Code##
This project utilizes various Go code examples for statistical analysis of Anscombe's quartet, focusing on automated, AI-assisted, and AI-generated methodologies. Some of the key code snippets include:
- **Linear Regression Calculation**: The `calculateLinearRegression(x, y []float64)` function is a key example of manually implemented code to ensure accuracy. It was improved over iterations with AI assistance, providing a clean and functional version for calculating the regression slope and intercept. This function can be found in the main project file `(main.go)`.
- **Descriptive Statistics Calculation**: The `printDescriptiveStats(x, y []float64)` function demonstrates how descriptive statistics, such as mean, variance, and correlation, are computed for each dataset in Anscombe's quartet. This code was initially generated with AI assistance and then modified for better functionality.
- **Unit Testing**: The `TestLinearRegression(t *testing.T)` function in the `main.go` file was developed to verify the correctness of the linear regression model. This code was partially generated using automated methods and AI-assisted tools to ensure accuracy and alignment with best practices in testing.
The code was obtained through a combination of automated code generation tools, AI-assisted programming, and manual tweaking to ensure compatibility with Go libraries and project requirements.

## Experiences with Methods ##
- **Automated Code Generation**: Automated tools were useful for generating basic structures and boilerplate code, such as setting up the Go environment and implementing simple functions. However, when it came to more complex tasks like linear regression calculations, the automated code often required extensive manual refinement. The rigidity of automated generation made it less effective for nuanced statistical computations.
- **AI-Assisted Programming**: AI-assisted programming provided significant benefits, especially in writing and refining complex functions. The suggestions from AI tools helped improve the performance and readability of the code, particularly in implementing calculations and debugging. This method saved time and ensured adherence to best practices without sacrificing accuracy.
- **AI-Generated Code**: The fully AI-generated code was often useful for quickly prototyping ideas and getting a rough draft. However, it generally required more modifications compared to AI-assisted suggestions, as it lacked contextual awareness of specific requirements, such as using correct data types or handling statistical calculations precisely. For example, in this assignment, ChatGpt can not solve the porblem of using linear regression function. 


