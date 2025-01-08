import React, { useState } from 'react';
import './AnalysisResult.css';

function AnalysisResult({ result }) {
  // Replace newlines with <br /> tags to preserve formatting
  const formattedResult = result.replace(/\n/g, '<br />');
  const [copyButtonText, setCopyButtonText] = useState('📋');

  const copyToClipboard = () => {
    navigator.clipboard.writeText(result).then(() => {
      setCopyButtonText('✔️'); // Show checkmark
      setTimeout(() => {
        setCopyButtonText('📋'); // Revert back to clipboard icon
      }, 1500);
    }, (err) => {
      console.error('Could not copy text: ', err);
    });
  };

  return (
    <div id="result" className="show">
      <button className="copy-button" onClick={copyToClipboard}>{copyButtonText}</button>
      <h2>Analysis Report</h2>
      <p dangerouslySetInnerHTML={{ __html: formattedResult }}></p>
    </div>
  );
}

export default AnalysisResult;
