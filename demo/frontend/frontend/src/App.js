import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [percentage, setPercentage] = useState(100);
  const [summary, setSummary] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    try {
      // Make the POST request to the Flask backend
      const response = await axios.post('http://localhost:5000/summarize', {
        text: text,
        percentage: percentage
      });
      setSummary(response.data.summary);
    } catch (error) {
      console.error('Error:', error);
      setSummary('Error in fetching summary');
    }
    setIsSubmitting(false);
  };

  return (
    <div>
      <h1>Text Summarization</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <textarea 
            value={text} 
            onChange={(e) => setText(e.target.value)} 
            placeholder="Enter text here"
            rows="10"
            cols="50"
          />
        </div>
        <div>
          <label>
            Percentage of information to retain: 
            <input 
              type="number" 
              value={percentage} 
              onChange={(e) => setPercentage(e.target.value)} 
              min="1"
              max="100"
            />
          </label>
        </div>
        <button type="submit" disabled={isSubmitting}>
          {isSubmitting ? 'Summarizing...' : 'Summarize'}
        </button>
      </form>
      {summary && (
        <div>
          <h3>Summary:</h3>
          <p>{summary}</p>
        </div>
      )}
    </div>
  );
}

export default App;
