import axios from 'axios';

export const analyzeComment = async (comment) => {
  const response = await axios.post('http://127.0.0.1:5000/analyze', { comment });
  return response.data;
};
