package main

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/montanaflynn/stats"
	"github.com/stretchr/testify/assert"
)

// Data for Anscombe's quartet
var datasets = []struct {
	x []float64
	y []float64
}{
	{
		x: []float64{10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0},
		y: []float64{8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68},
	},
	{
		x: []float64{10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0},
		y: []float64{9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74},
	},
	{
		x: []float64{10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0},
		y: []float64{7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73},
	},
	{
		x: []float64{8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0},
		y: []float64{6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.5, 5.56, 7.91, 6.89},
	},
}

func main() {
	for i, set := range datasets {
		fmt.Printf("Dataset %d Descriptive Statistics:\n", i+1)
		printDescriptiveStats(set.x, set.y)
		start := time.Now()
		slope, intercept, err := calculateLinearRegression(set.x, set.y)
		if err != nil {
			fmt.Printf("Error in calculating regression for Set %d: %v\n", i+1, err)
			continue
		}
		elapsed := time.Since(start)
		fmt.Printf("Set %d: Slope: %.2f, Intercept: %.2f, Time taken: %s\n", i+1, slope, intercept, elapsed)

		// Calculate and print residuals and R-squared value
		residuals := calculateResiduals(set.x, set.y, slope, intercept)
		rSquared := calculateRSquared(set.y, residuals)
		fmt.Printf("Set %d: R-squared: %.2f\n\n", i+1, rSquared)
	}
}

func printDescriptiveStats(x, y []float64) {
	meanX, _ := stats.Mean(x)
	meanY, _ := stats.Mean(y)
	varX, _ := stats.Variance(x)
	varY, _ := stats.Variance(y)
	stdDevX, _ := stats.StandardDeviation(x)
	stdDevY, _ := stats.StandardDeviation(y)
	correlation, _ := stats.Correlation(x, y)

	fmt.Printf("Mean of x: %.2f, Mean of y: %.2f\n", meanX, meanY)
	fmt.Printf("Variance of x: %.2f, Variance of y: %.2f\n", varX, varY)
	fmt.Printf("Standard Deviation of x: %.2f, Standard Deviation of y: %.2f\n", stdDevX, stdDevY)
	fmt.Printf("Correlation between x and y: %.2f\n\n", correlation)
}

func calculateLinearRegression(x, y []float64) (float64, float64, error) {
	if len(x) != len(y) {
		return 0, 0, fmt.Errorf("x and y must have the same length")
	}
	if len(x) == 0 {
		return 0, 0, fmt.Errorf("x and y must not be empty")
	}

	meanX, _ := stats.Mean(x)
	meanY, _ := stats.Mean(y)

	var numerator, denominator float64
	for i := 0; i < len(x); i++ {
		numerator += (x[i] - meanX) * (y[i] - meanY)
		denominator += (x[i] - meanX) * (x[i] - meanX)
	}

	if denominator == 0 {
		return 0, 0, fmt.Errorf("denominator is zero, cannot compute slope")
	}

	slope := numerator / denominator
	intercept := meanY - slope*meanX

	return slope, intercept, nil
}
func calculateResiduals(x, y []float64, slope, intercept float64) []float64 {
	residuals := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		predictedY := slope*x[i] + intercept
		residuals[i] = y[i] - predictedY
	}
	return residuals
}

func calculateRSquared(y, residuals []float64) float64 {
	sst := 0.0
	meanY, _ := stats.Mean(y)
	for _, yi := range y {
		sst += math.Pow(yi-meanY, 2)
	}

	ssr := 0.0
	for _, res := range residuals {
		ssr += math.Pow(res, 2)
	}
	return 1 - (ssr / sst)
}

func TestLinearRegression(t *testing.T) {
	expectedResults := []struct {
		slope     float64
		intercept float64
		rSquared  float64
	}{
		{slope: 0.500, intercept: 3.000, rSquared: 0.67},
		{slope: 0.500, intercept: 3.001, rSquared: 0.67},
		{slope: 0.499, intercept: 3.002, rSquared: 0.67},
		{slope: 0.000, intercept: 5.000, rSquared: 0.00},
	}

	for i, set := range datasets {
		slope, intercept, err := calculateLinearRegression(set.x, set.y)
		if err != nil {
			t.Errorf("Error in calculating regression for Set %d: %v", i+1, err)
			continue
		}
		assert.InDelta(t, expectedResults[i].slope, slope, 0.01, fmt.Sprintf("Slope mismatch for Set %d", i+1))
		assert.InDelta(t, expectedResults[i].intercept, intercept, 0.01, fmt.Sprintf("Intercept mismatch for Set %d", i+1))

		// Calculate residuals and R-squared value
		residuals := calculateResiduals(set.x, set.y, slope, intercept)
		rSquared := calculateRSquared(set.y, residuals)
		assert.InDelta(t, expectedResults[i].rSquared, rSquared, 0.01, fmt.Sprintf("R-squared mismatch for Set %d", i+1))

		fmt.Printf("Residuals for Set %d: %v\n\n", i+1, residuals)
	}
}
