// src/components/LoadingBar.js
import React from 'react';
import './LoadingBar.css'

function LoadingBar({ progress }) {
  return (
    <div id="loading-bar-container">
      <div id="loading-bar" style={{ width: `${progress}%` }}></div>
    </div>
  );
}

export default LoadingBar;
