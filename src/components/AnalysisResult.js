import React from 'react';
import './AnalysisResult.css';

function AnalysisResult({ result }) {
  // Replace newlines with <br /> tags to preserve formatting
  const formattedResult = result.replace(/\n/g, '<br />');

  return (
    <div id="result" className="show">
      <h2>Analysis Report</h2>
      <p dangerouslySetInnerHTML={{ __html: formattedResult }}></p>
    </div>
  );
}

export default AnalysisResult;
