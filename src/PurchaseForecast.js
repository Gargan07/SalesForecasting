import React, { useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import Papa from "papaparse";
import { Chart, registerables } from "chart.js";

Chart.register(...registerables);

let chartInstance = null; // Track the chart instance globally

const PurchaseForecast = () => {
  const [data, setData] = useState([]);
  const [productList, setProductList] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [preprocess, setPreprocess] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const chartRef = useRef(null);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (result) => {
          setData(result.data);
          const uniqueProducts = getUniqueProducts(result.data);
          setProductList(uniqueProducts);
        },
        error: (error) => console.error("Error parsing CSV:", error),
      });
    }
  };

  const getUniqueProducts = (data) => {
    const productsSet = new Set();
    data.forEach((row) => {
      if (row.short_desc) 
        productsSet.add(row.short_desc);
        
    });
    return Array.from(productsSet).sort();
  };

  const filterData = () => {
    console.log("Selected Product:", selectedProduct);
    return selectedProduct
      ? data.filter((entry) => entry.short_desc === selectedProduct)
      : data;
  };

  const preprocessData = (data) => {
    const productDescriptionMap = {};
    let productIndex = 0; 

    const quantities = data.map((item) => {
      const quantity = parseFloat(item.total_sold);
      if (isNaN(quantity)) {
        console.warn(`Invalid quantity for item with ID ${item.id}, skipping.`);
        return null;
      }
      return quantity;
    }).filter((quantity) => quantity !== null);

    const minQuantity = Math.min(...quantities);
    const maxQuantity = Math.max(...quantities);

    const processedData = preprocess.length ? preprocess : data.map((item) => {
      console.log("Item:", item);
      const salesDateParts = item.created.split("/");
      const salesMonth = salesDateParts.length === 2 ? parseInt(salesDateParts[1], 10) : null;
  
      if (isNaN(salesMonth)) {
        console.warn(`Invalid date format for item with ID ${item.id}, skipping.`);
        return null; // Skip invalid date items
      }
  
      // Product encoding to prevent product name duplicates
      const productEncoded =
        productDescriptionMap[item.short_desc] !== undefined
          ? productDescriptionMap[item.short_desc]
          : (productDescriptionMap[item.short_desc] = productIndex++);
  
          const normalizedQuantity =
          data.every((item) => item.total_sold === data[0].total_sold)
            ? 0
            : (parseFloat(item.total_sold) - minQuantity) / (maxQuantity - minQuantity);
        
      return {
        sales_date: salesMonth,
        product_description: productEncoded,
        quantity_sold: normalizedQuantity,
      };
    }).filter(item => item !== null); // Remove any null items from the array
    return { processedData, minQuantity, maxQuantity };
  };

  const prepareData = () => {
    const filteredData = filterData();
    const { processedData, minQuantity, maxQuantity } = preprocessData(filteredData);
    const quantities = processedData.map((entry) => entry.quantity_sold);
    const xs = tf.tensor2d(
      quantities.slice(0, -1).map((_, index) => [
        processedData[index].sales_date,
        processedData[index].product_description
      ]),
      [quantities.length - 1, 2]
    );
    const ys = tf.tensor1d(quantities.slice(1));

    console.log("Quantities:", quantities);
    
    
    return { xs, ys, minQuantity, maxQuantity };
  };

const trainModel = async () => {
  const { xs, ys, minQuantity, maxQuantity } = prepareData();

  // Build the model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, inputShape: [2], activation: "relu" })); // Input layer
  model.add(tf.layers.dense({ units: 32, activation: "relu" })); // Hidden layer
  model.add(tf.layers.dense({ units: 1, activation: "linear" })); // Output layer

  // Compile the model with Adam optimizer and MSE loss function
  model.compile({ optimizer: tf.train.adam(0.01), loss: "meanSquaredError" });

  // Fit the model and train
  await model.fit(xs, ys, {
    epochs: 200,
    callbacks: [
      {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch: ${epoch + 1}, Loss: ${logs.loss}`);
        },
      },
    ],
  });

  // Return the trained model and normalization parameters
  return { model, minQuantity, maxQuantity };
};

const predictNextQuantity = async () => {
  const { model, minQuantity, maxQuantity } = await trainModel();

  const filteredData = filterData();
  const { processedData } = preprocessData(filteredData);
  setPreprocess(processedData);

  console.log("Filtered Data:", filteredData);
  console.log("Processed Data:", processedData);

  const lastDataPoint = processedData[processedData.length - 1];
  const lastSalesDate = lastDataPoint.sales_date;
  const lastProductDescription = lastDataPoint.product_description;
  const forecastedQuantities = [];

  for (let i = 1; i <= 6; i++) {
    const inputTensor = tf.tensor2d([[lastSalesDate + i, lastProductDescription]]);
    const outputTensor = model.predict(inputTensor);
    const normalizedPrediction = outputTensor.dataSync()[0];
    let denormalizedPrediction =
      normalizedPrediction * (maxQuantity - minQuantity) + minQuantity;
    denormalizedPrediction = Math.max(0, denormalizedPrediction);
    forecastedQuantities.push(denormalizedPrediction.toFixed(2));
  }
  setPrediction(forecastedQuantities);
  console.log("Prediction:", forecastedQuantities);
  renderChart(filteredData.map((d) => parseFloat(d.total_sold)), forecastedQuantities);
  setPreprocess([]);
};

const renderChart = (actualSales, predictedSales) => {
  const ctx = chartRef.current.getContext("2d");

  if (chartInstance) {
    chartInstance.destroy(); // Destroy the old chart instance
  }

  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      // x-axis labels starting from 0 (for the "Month 0") but actual sales start from 1
      labels: ["0", ...Array.from({ length: actualSales.length }, (_, i) => (i + 1).toString()), ...Array.from({ length: predictedSales.length }, (_, i) => (i + actualSales.length + 1).toString())],
      
      datasets: [
        {
          label: "Actual Sales",
          // The actualSales will have a null value at the first position to align the data starting from Month 1
          data: [null, ...actualSales], 
          borderColor: "blue",
          borderWidth: 2,
          fill: false,
          tension: 0.2,
        },
        {
          label: "Predicted Sales",
          // For predicted sales, it starts after actual sales, so it's padded with nulls at the start
          data: [...new Array(actualSales.length + 1).fill(null), ...predictedSales],
          borderColor: "orange",
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: "Actual vs. Predicted Sales",
        },
        legend: {
          position: "top",
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Months" },
          min: 0, // Start the x-axis from 0
        },
        y: {
          title: { display: true, text: "Sales Quantity" },
        },
      },
    },
  });
};


return (
  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "20px" }}>
    <h1>Sales Prediction</h1>
    <input type="file" accept=".csv" onChange={handleFileUpload} style={{ marginLeft: "30px", marginBottom: "10px" }}/>
    {productList.length > 0 && (
      <select onChange={(e) => setSelectedProduct(e.target.value)} style={{ marginBottom: "10px" }}>
        <option value="">Select a product</option>
        {productList.map((product, index) => (
          <option key={index} value={product}>
            {product}
          </option>
        ))}
      </select>
    )}
    <button onClick={predictNextQuantity}>Predict</button>
    <canvas ref={chartRef} width="500" height="400"></canvas>
  </div>
);
};

export default PurchaseForecast;
