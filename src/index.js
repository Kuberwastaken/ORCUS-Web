import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // Main app component

// Create a root and render the App component inside the root div element
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
