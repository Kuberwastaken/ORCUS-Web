import React, { useState } from 'react';
import './CommentForm.css';

function CommentForm({ onAnalyze }) {
  const [comment, setComment] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onAnalyze(comment);
  };

  return (
    <div className="textarea-container">
      <form onSubmit={handleSubmit}>
        <textarea
          id="comment"
          placeholder="Paste a LinkedIn comment here..."
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        ></textarea>
        <button type="submit">Analyze Comment</button>
      </form>
    </div>
  );
}

export default CommentForm;
