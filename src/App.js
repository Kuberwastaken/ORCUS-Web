// App.js
import React, { useState } from 'react';
import CommentForm from './components/CommentForm';
import LoadingBar from './components/LoadingBar';
import AnalysisResult from './components/AnalysisResult';
import { analyzeComment } from './api/analyze';
import './App.css';

function App() {
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleAnalyze = async (comment) => {
    setLoading(true);
    setProgress(0);
    setResult('');

    // Slower progress increment
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev < 30) {  // Reduced from 90 to 85
          return prev + 0.5;  // Reduced from 2 to 0.5
        }
        return prev;
      });
    }, 170);  // Increased from 100 to 200

    try {
      const data = await analyzeComment(comment);
      clearInterval(interval);
      
      // Gradual progress to 100%
      setProgress(50);
      setTimeout(() => {
        setProgress(69);
      }, 600);
      setTimeout(() => {
        setProgress(84);
        setResult(data.funny_comment);
      }, 600);
      setTimeout(() => {
        setProgress(100);
        setResult(data.funny_comment);
      }, 600);
      
      // Longer delay before hiding loading bar
      setTimeout(() => {
        setLoading(false);
      }, 1000);

    } catch (error) {
      console.error('Error analyzing comment:', error);
      clearInterval(interval);
      setLoading(false);
      setProgress(0);
    }
  };

  return (
    <div className="container">
      <h1>ORCUS</h1>
      <p id="acronym">
        <span className="acronym-letter">O</span>bservational&nbsp;&nbsp;
        <span className="acronym-letter">R</span>ecognition
        <span className="line-break"></span>&nbsp;&nbsp;of&nbsp;&nbsp;
        <span className="acronym-letter">C</span>ontent&nbsp;with&nbsp;&nbsp;
        <span className="acronym-letter">U</span>nnatural&nbsp;&nbsp;
        <span className="acronym-letter">S</span>peech
      </p>

      <CommentForm onAnalyze={handleAnalyze} />
      {loading && <LoadingBar progress={progress} />}
      {result && <AnalysisResult result={result} />}
    </div>
  );
}

export default App;