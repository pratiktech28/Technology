import React, { useState } from 'react';
import { Upload, BarChart3, Brain, Download, TrendingUp, AlertCircle, CheckCircle, Zap } from 'lucide-react';

const DataIntelligencePlatform = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
    }
  };

  const analyzeData = async () => {
    setAnalyzing(true);
    
    // Simulate AI analysis
    setTimeout(() => {
      setResults({
        insights: [
          { title: 'Data Quality', score: 92, status: 'good' },
          { title: 'Missing Values', score: 8, status: 'good' },
          { title: 'Outliers Detected', score: 15, status: 'warning' },
          { title: 'Feature Correlation', score: 85, status: 'good' }
        ],
        predictions: {
          accuracy: 94.5,
          confidence: 89.2,
          samples: 1250
        },
        recommendations: [
          'Consider feature engineering for better model performance',
          'Apply normalization to improve convergence',
          'Remove 15 outlier samples for cleaner training data'
        ]
      });
      setAnalyzing(false);
      setActiveTab('results');
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-2 rounded-lg">
                <Brain className="text-white" size={28} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">AI Data Intelligence</h1>
                <p className="text-sm text-gray-600">Automated Insights & Analytics Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm">
              <Zap className="text-yellow-500" size={16} />
              <span className="text-gray-700 font-medium">Powered by ML</span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="bg-white rounded-xl shadow-sm p-2 flex space-x-2">
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex-1 px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'upload'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <Upload className="inline mr-2" size={18} />
            Upload Data
          </button>
          <button
            onClick={() => setActiveTab('preprocess')}
            className={`flex-1 px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'preprocess'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <BarChart3 className="inline mr-2" size={18} />
            Preprocess
          </button>
          <button
            onClick={() => setActiveTab('results')}
            className={`flex-1 px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'results'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <TrendingUp className="inline mr-2" size={18} />
            Results
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 pb-12">
        {activeTab === 'upload' && (
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Your Dataset</h2>
            
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-blue-500 transition-colors cursor-pointer bg-gray-50">
              <input
                type="file"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
                accept=".csv,.xlsx,.json"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <Upload className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-lg font-medium text-gray-700 mb-2">
                  {file ? file.name : 'Click to upload or drag and drop'}
                </p>
                <p className="text-sm text-gray-500">CSV, XLSX, or JSON (Max 50MB)</p>
              </label>
            </div>

            {file && (
              <div className="mt-8 space-y-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <CheckCircle className="text-blue-600" size={24} />
                    <div>
                      <p className="font-medium text-gray-900">{file.name}</p>
                      <p className="text-sm text-gray-600">{(file.size / 1024).toFixed(2)} KB</p>
                    </div>
                  </div>
                </div>

                <button
                  onClick={analyzeData}
                  className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 rounded-xl font-semibold text-lg hover:shadow-lg transition-all"
                >
                  Start AI Analysis
                </button>
              </div>
            )}
          </div>
        )}

        {activeTab === 'preprocess' && (
          <div className="bg-white rounded-2xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Data Preprocessing</h2>
            
            <div className="space-y-6">
              <div className="border border-gray-200 rounded-xl p-6">
                <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
                  <CheckCircle className="text-green-600 mr-2" size={20} />
                  Cleaning & Validation
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-600">Missing Values</p>
                    <p className="text-2xl font-bold text-gray-900">8%</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-600">Duplicates Removed</p>
                    <p className="text-2xl font-bold text-gray-900">23</p>
                  </div>
                </div>
              </div>

              <div className="border border-gray-200 rounded-xl p-6">
                <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
                  <CheckCircle className="text-green-600 mr-2" size={20} />
                  Normalization Applied
                </h3>
                <p className="text-gray-600">StandardScaler normalization applied to numerical features</p>
              </div>

              <div className="border border-gray-200 rounded-xl p-6">
                <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
                  <CheckCircle className="text-green-600 mr-2" size={20} />
                  Feature Engineering
                </h3>
                <p className="text-gray-600">15 new features generated from existing data</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'results' && (
          <div className="space-y-6">
            {analyzing ? (
              <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
                <p className="text-lg font-medium text-gray-700">Analyzing your data...</p>
                <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
              </div>
            ) : results ? (
              <>
                {/* Insights Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {results.insights.map((insight, idx) => (
                    <div key={idx} className="bg-white rounded-xl shadow-lg p-6">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="font-semibold text-gray-900">{insight.title}</h3>
                        {insight.status === 'good' ? (
                          <CheckCircle className="text-green-600" size={20} />
                        ) : (
                          <AlertCircle className="text-yellow-600" size={20} />
                        )}
                      </div>
                      <p className="text-3xl font-bold text-gray-900">{insight.score}%</p>
                      <div className="mt-3 bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            insight.status === 'good' ? 'bg-green-600' : 'bg-yellow-600'
                          }`}
                          style={{ width: `${insight.score}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Model Performance */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">Model Performance</h2>
                  <div className="grid grid-cols-3 gap-6">
                    <div className="text-center">
                      <p className="text-gray-600 mb-2">Accuracy</p>
                      <p className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                        {results.predictions.accuracy}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-600 mb-2">Confidence</p>
                      <p className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                        {results.predictions.confidence}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-600 mb-2">Samples</p>
                      <p className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                        {results.predictions.samples}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-white rounded-2xl shadow-lg p-8">
                  <h2 className="text-2xl font-bold text-gray-900 mb-6">AI Recommendations</h2>
                  <div className="space-y-3">
                    {results.recommendations.map((rec, idx) => (
                      <div key={idx} className="flex items-start space-x-3 p-4 bg-blue-50 rounded-lg">
                        <TrendingUp className="text-blue-600 mt-1 flex-shrink-0" size={20} />
                        <p className="text-gray-700">{rec}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Export */}
                <button className="w-full bg-gradient-to-r from-green-600 to-emerald-600 text-white py-4 rounded-xl font-semibold text-lg hover:shadow-lg transition-all flex items-center justify-center space-x-2">
                  <Download size={20} />
                  <span>Export Full Report (PDF)</span>
                </button>
              </>
            ) : (
              <div className="bg-white rounded-2xl shadow-lg p-12 text-center">
                <AlertCircle className="text-gray-400 mx-auto mb-4" size={48} />
                <p className="text-lg text-gray-600">No results yet. Upload and analyze data first.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DataIntelligencePlatform;
