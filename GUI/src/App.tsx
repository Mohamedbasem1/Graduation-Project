import React, { useState } from 'react';
import { Send, Loader2, Book, MessageCircle, AlertCircle } from 'lucide-react';

// Mock ML model response - replace this with your actual API call
const mockMlRequest = async (text: string): Promise<string[]> => {
  try {
    const response = await fetch('http://localhost:5000/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: text }),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    return data.answers; // Assuming the backend returns an object with an 'answers' field
  } catch (error) {
    console.error('Error fetching model response:', error);
    throw error; // Re-throw the error to handle it in the component
  }
};

function App() {
  const [inputText, setInputText] = useState('');
  const [results, setResults] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const mlResults = await mockMlRequest(inputText);
      setResults(mlResults); // Set the results state with the list of answers
    } catch (error) {
      setError('An error occurred while processing your question. Please try again.');
      console.error('Error processing question:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f9fa]">
      {/* Header */}
      <header className="bg-[#004d40] text-white py-6 shadow-lg">
        <div className="max-w-4xl mx-auto px-4">
          <div className="flex items-center gap-3">
            <Book className="w-8 h-8" />
            <h1 className="text-3xl font-bold">Quran Q&A Assistant</h1>
          </div>
          <p className="mt-2 text-emerald-100">Ask questions about the Holy Quran and Islamic teachings</p>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Question Form */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-8">
          <form onSubmit={handleSubmit}>
            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full p-4 border border-emerald-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none h-32 text-gray-800"
                placeholder="Type your question about the Quran..."
                aria-label="Type your question about the Quran"
              />
              <button
                type="submit"
                disabled={isLoading || !inputText.trim()}
                className="absolute bottom-4 right-4 bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                aria-label="Submit question"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    <span>Ask</span>
                  </>
                )}
              </button>
            </div>
          </form>
          {error && (
            <div className="mt-4 flex items-center gap-2 text-red-600">
              <AlertCircle className="w-5 h-5" />
              <p>{error}</p>
            </div>
          )}
        </div>

        {/* Results */}
        {results.length > 0 && (
          <div className="bg-white rounded-xl shadow-md p-6">
            <div className="flex items-center gap-2 mb-6">
              <MessageCircle className="w-6 h-6 text-emerald-600" />
              <h2 className="text-xl font-semibold text-gray-800">Answers</h2>
            </div>
            <ul className="space-y-4">
              {results.map((result, index) => (
                <li
                  key={index}
                  className="p-4 bg-emerald-50 rounded-lg border border-emerald-100 text-gray-700 hover:bg-emerald-100 transition-colors"
                >
                  <div className="flex gap-3">
                    <span className="text-emerald-600 font-semibold">{index + 1}.</span>
                    <p>{result}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-[#004d40] text-emerald-100 py-4 mt-auto">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p>Use this tool responsibly and verify answers with authentic Islamic sources</p>
        </div>
      </footer>
    </div>
  );
}

export default App;