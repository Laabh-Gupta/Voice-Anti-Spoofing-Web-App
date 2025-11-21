// In src/App.js

import React, { useState } from 'react';
import './App.css'; // We'll use this for some basic styling

function App() {
  // State variables to manage the file, prediction, and loading status
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // This function is called when the user selects a file
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction(null); // Reset previous prediction
    setError(null); // Reset previous error
  };

  // This function is called when the user clicks the "Classify Audio" button
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }

    setIsLoading(true); // Show a loading message
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Send the file to your FastAPI backend
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Something went wrong with the API call.');
      }

      const data = await response.json();
      setPrediction(data); // Store the prediction result
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false); // Hide the loading message
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎙️ Voice Anti-Spoofing System</h1>
        <p>Upload an audio file (.wav or .mp3) to see if it's real or AI-generated.</p>
        
        <form onSubmit={handleSubmit}>
          <input type="file" onChange={handleFileChange} accept=".wav,.mp3" />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Analyzing...' : 'Classify Audio'}
          </button>
        </form>

        {/* Conditionally display the loading message, error, or prediction */}
        {isLoading && <p>Loading...</p>}
        {error && <p className="error">Error: {error}</p>}
        {prediction && (
          <div className="result">
            <h2>Prediction Result:</h2>
            <p><strong>Filename:</strong> {prediction.filename}</p>
            <p className={`prediction-${prediction.predicted_class}`}>
              <strong>Predicted Class:</strong> {prediction.predicted_class.toUpperCase()}
            </p>
            <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;