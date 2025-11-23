// src/App.js
import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Handle file selection
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction(null);
    setError(null);
  };

  // Submit audio to backend
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(
        `${process.env.REACT_APP_BACKEND_URL}/predict/`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("API Error: Unable to fetch prediction.");
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎙️ Voice Anti-Spoofing System</h1>
        <p>Upload a WAV file to detect if it's REAL or AI-GENERATED.</p>

        {/* Upload Form */}
        <form onSubmit={handleSubmit}>
          <input type="file" accept=".wav,.mp3,audio/*" onChange={handleFileChange} />
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Analyzing..." : "Classify Audio"}
          </button>
        </form>

        {/* Loading */}
        {isLoading && <p className="loading">Processing audio...</p>}

        {/* Error */}
        {error && <p className="error">❌ Error: {error}</p>}

        {/* API Error */}
        {prediction && prediction.error && (
          <p className="error">❌ Error: {prediction.error}</p>
        )}

        {/* SUCCESS RESULT */}
        {prediction && prediction.predicted_class && (
          <div className="result">
            <h2>Prediction Result</h2>
            <p>
              <strong>Filename:</strong> {prediction.filename}
            </p>
            <p>
              <strong>Predicted Class:</strong>{" "}
              {prediction.predicted_class.toUpperCase()}
            </p>
            <p>
              <strong>Confidence:</strong>{" "}
              {(prediction.confidence * 100).toFixed(2)}%
            </p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
