import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, 
  Upload, 
  FileText, 
  AlertTriangle, 
  Clock, 
  Database,
  Trash2,
  CheckCircle,
  AlertCircle,
  BookOpen,
  Scale,
  Gavel
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const LegalRAGApp = () => {
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [activeTab, setActiveTab] = useState('search');
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const fileInputRef = useRef(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`);
      if (response.ok) {
        const docs = await response.json();
        setDocuments(docs);
      }
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  const handleFileSelect = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
  };

  const uploadDocuments = async () => {
    if (selectedFiles.length === 0) return;

    setUploading(true);
    
    for (const file of selectedFiles) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/upload-document`, {
          method: 'POST',
          body: formData,
        });
        
        if (response.ok) {
          console.log(`${file.name} uploaded successfully`);
        } else {
          console.error(`Error uploading ${file.name}`);
        }
      } catch (error) {
        console.error(`Error uploading ${file.name}:`, error);
      }
    }
    
    setSelectedFiles([]);
    setUploading(false);
    await fetchDocuments();
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          document_ids: selectedDocuments.length > 0 ? selectedDocuments : null,
          max_results: 10,
          similarity_threshold: 0.6
        }),
      });

      if (response.ok) {
        const results = await response.json();
        setSearchResults(results);
      } else {
        console.error('Search failed');
      }
    } catch (error) {
      console.error('Error during search:', error);
    }
    setLoading(false);
  };

  const deleteDocument = async (documentId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        await fetchDocuments();
        setSelectedDocuments(prev => prev.filter(id => id !== documentId));
      }
    } catch (error) {
      console.error('Error deleting document:', error);
    }
  };

  const toggleDocumentSelection = (documentId) => {
    setSelectedDocuments(prev => 
      prev.includes(documentId) 
        ? prev.filter(id => id !== documentId)
        : [...prev, documentId]
    );
  };

  const getDocumentTypeIcon = (type) => {
    switch (type) {
      case 'contract':
        return <FileText className="w-4 h-4 text-blue-600" />;
      case 'case_law':
        return <Gavel className="w-4 h-4 text-purple-600" />;
      case 'statute':
        return <Scale className="w-4 h-4 text-green-600" />;
      default:
        return <BookOpen className="w-4 h-4 text-gray-600" />;
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Scale className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Legal Research Assistant</h1>
                <p className="text-sm text-gray-500">Multi-Document RAG System</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Database className="w-4 h-4" />
              <span>{documents.length} documents</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
     {/* Tab Navigation */}
<div className="flex space-x-1 bg-gray-100 p-1 rounded-lg mb-8 w-fit">
  <button
    onClick={() => setActiveTab('search')}
    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
      activeTab === 'search'
        ? 'bg-white text-blue-600 shadow-sm'
        : 'text-gray-600 hover:text-gray-900'
    }`}
  >
    <Search className="w-4 h-4 inline mr-2" />
    Search Documents
  </button>
  <button
    onClick={() => setActiveTab('upload')}
    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
      activeTab === 'upload'
        ? 'bg-white text-blue-600 shadow-sm'
        : 'text-gray-600 hover:text-gray-900'
    }`}
  >
    <Upload className="w-4 h-4 inline mr-2" />
    Upload Documents
  </button>
  <button
    onClick={() => setActiveTab('manage')}
    className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
      activeTab === 'manage'
        ? 'bg-white text-blue-600 shadow-sm'
        : 'text-gray-600 hover:text-gray-900'
    }`}
  >
    <FileText className="w-4 h-4 inline mr-2" />
    Manage Documents
  </button>
</div>


       {/* Search Tab */}
       {activeTab === 'search' && (
         <div className="space-y-6">
           {/* Search Interface */}
           <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
             <div className="space-y-4">
               <div>
                 <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                   Legal Query
                 </label>
                 <div className="relative">
                   <textarea
                     id="query"
                     rows={3}
                     className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                     placeholder="Enter your legal question (e.g., 'What are the termination clauses in the employment contracts?', 'Define force majeure provisions', 'What are the indemnification requirements?')"
                     value={query}
                     onChange={(e) => setQuery(e.target.value)}
                   />
                   <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                     <Search className="w-5 h-5 text-gray-400" />
                   </div>
                 </div>
               </div>

               {/* Document Filter */}
               {documents.length > 0 && (
                 <div>
                   <label className="block text-sm font-medium text-gray-700 mb-2">
                     Search Scope (Optional)
                   </label>
                   <div className="flex flex-wrap gap-2">
                     <button
                       onClick={() => setSelectedDocuments([])}
                       className={`px-3 py-1 rounded-full text-xs font-medium ${
                         selectedDocuments.length === 0
                           ? 'bg-blue-100 text-blue-800'
                           : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                       }`}
                     >
                       All Documents
                     </button>
                     {documents.slice(0, 5).map((doc) => (
                       <button
                         key={doc.document_id}
                         onClick={() => toggleDocumentSelection(doc.document_id)}
                         className={`px-3 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${
                           selectedDocuments.includes(doc.document_id)
                             ? 'bg-blue-100 text-blue-800'
                             : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                         }`}
                       >
                         {getDocumentTypeIcon(doc.document_type)}
                         <span className="truncate max-w-24">{doc.filename}</span>
                       </button>
                     ))}
                   </div>
                 </div>
               )}

               <button
                 onClick={handleSearch}
                 disabled={!query.trim() || loading}
                 className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
               >
                 {loading ? (
                   <>
                     <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                     <span>Searching...</span>
                   </>
                 ) : (
                   <>
                     <Search className="w-4 h-4" />
                     <span>Search Legal Documents</span>
                   </>
                 )}
               </button>
             </div>
           </div>

           {/* Search Results */}
           {searchResults && (
             <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
               <div className="flex items-center justify-between mb-6">
                 <h2 className="text-lg font-semibold text-gray-900">Search Results</h2>
                 <div className="flex items-center space-x-4 text-sm text-gray-500">
                   <div className="flex items-center space-x-1">
                     <Clock className="w-4 h-4" />
                     <span>{searchResults.processing_time.toFixed(2)}s</span>
                   </div>
                   <div className="flex items-center space-x-1">
                     <FileText className="w-4 h-4" />
                     <span>{searchResults.citations.length} citations</span>
                   </div>
                 </div>
               </div>

               {/* Main Answer */}
               <div className="mb-6">
                 <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                   <h3 className="font-medium text-blue-900 mb-2">Legal Analysis</h3>
                   <div className="text-blue-800 whitespace-pre-line">
                     {searchResults.answer}
                   </div>
                 </div>
               </div>

               {/* Conflicts Alert */}
               {searchResults.conflicts && (
                 <div className="mb-6">
                   <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                     <div className="flex items-start space-x-2">
                       <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5" />
                       <div>
                         <h3 className="font-medium text-amber-900 mb-2">Potential Conflicts Detected</h3>
                         <p className="text-amber-800 text-sm mb-3">{searchResults.conflicts.analysis}</p>
                         <div className="text-xs text-amber-700">
                           Confidence: {(searchResults.conflicts.confidence * 100).toFixed(0)}%
                         </div>
                       </div>
                     </div>
                   </div>
                 </div>
               )}

               {/* Citations */}
               <div>
                 <h3 className="font-medium text-gray-900 mb-4">Legal Citations</h3>
                 <div className="space-y-4">
                   {searchResults.citations.map((citation, index) => (
                     <div key={index} className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors">
                       <div className="flex items-start justify-between mb-2">
                         <div className="flex items-center space-x-2">
                           {getDocumentTypeIcon(citation.document_type)}
                           <h4 className="font-medium text-gray-900">{citation.document_name}</h4>
                           <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                             {citation.document_type.replace('_', ' ')}
                           </span>
                         </div>
                         <div className="flex items-center space-x-2">
                           <div className="text-xs text-gray-500">
                             Relevance: {(citation.relevance_score * 100).toFixed(0)}%
                           </div>
                           <div className={`w-2 h-2 rounded-full ${
                             citation.relevance_score > 0.8 ? 'bg-green-500' :
                             citation.relevance_score > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                           }`}></div>
                         </div>
                       </div>
                       <div className="text-sm text-gray-600 mb-2">
                         <strong>Section:</strong> {citation.section}
                       </div>
                       <div className="text-sm text-gray-800 bg-gray-50 p-3 rounded border-l-4 border-blue-500">
                         "{citation.text_snippet}"
                       </div>
                     </div>
                   ))}
                 </div>
               </div>
             </div>
           )}
         </div>
       )}

       {/* Upload Tab */}
       {activeTab === 'upload' && (
         <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
           <div className="text-center">
             <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
               <Upload className="w-8 h-8 text-blue-600" />
             </div>
             <h2 className="text-lg font-semibold text-gray-900 mb-2">Upload Legal Documents</h2>
             <p className="text-gray-600 mb-6">
               Support for PDF, DOCX, and TXT files. The system will automatically detect document types and create intelligent chunks.
             </p>
             
             <div className="max-w-md mx-auto">
               <input
                 ref={fileInputRef}
                 type="file"
                 multiple
                 accept=".pdf,.docx,.txt"
                 onChange={handleFileSelect}
                 className="hidden"
                 id="file-upload"
               />
               <label
                 htmlFor="file-upload"
                 className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors"
               >
                 <Upload className="w-8 h-8 text-gray-400 mb-2" />
                 <span className="text-sm text-gray-600">Click to select files</span>
                 <span className="text-xs text-gray-500">PDF, DOCX, TXT</span>
               </label>

               {selectedFiles.length > 0 && (
                 <div className="mt-4 space-y-2">
                   <h3 className="text-sm font-medium text-gray-900">Selected Files:</h3>
                   {selectedFiles.map((file, index) => (
                     <div key={index} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                       <div className="flex items-center space-x-2">
                         <FileText className="w-4 h-4 text-gray-500" />
                         <span className="text-sm text-gray-700">{file.name}</span>
                       </div>
                       <span className="text-xs text-gray-500">
                         {(file.size / 1024 / 1024).toFixed(2)} MB
                       </span>
                     </div>
                   ))}
                   <button
                     onClick={uploadDocuments}
                     disabled={uploading}
                     className="w-full mt-4 bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                   >
                     {uploading ? (
                       <>
                         <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                         <span>Uploading...</span>
                       </>
                     ) : (
                       <>
                         <CheckCircle className="w-4 h-4" />
                         <span>Upload Documents</span>
                       </>
                     )}
                   </button>
                 </div>
               )}
             </div>
           </div>
         </div>
       )}

       {/* Manage Documents Tab */}
       {activeTab === 'manage' && (
         <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
           <div className="flex items-center justify-between mb-6">
             <h2 className="text-lg font-semibold text-gray-900">Document Library</h2>
             <div className="text-sm text-gray-500">
               {documents.length} document{documents.length !== 1 ? 's' : ''} total
             </div>
           </div>

           {documents.length === 0 ? (
             <div className="text-center py-12">
               <FileText className="mx-auto w-12 h-12 text-gray-400 mb-4" />
               <h3 className="text-lg font-medium text-gray-900 mb-2">No documents uploaded</h3>
               <p className="text-gray-600 mb-4">Get started by uploading your first legal document.</p>
               <button
                 onClick={() => setActiveTab('upload')}
                 className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
               >
                 Upload Documents
               </button>
             </div>
           ) : (
             <div className="grid gap-4">
               {documents.map((doc) => (
                 <div key={doc.document_id} className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors">
                   <div className="flex items-center justify-between">
                     <div className="flex items-center space-x-3">
                       {getDocumentTypeIcon(doc.document_type)}
                       <div>
                         <h3 className="font-medium text-gray-900">{doc.filename}</h3>
                         <div className="flex items-center space-x-4 text-sm text-gray-500 mt-1">
                           <span className="capitalize">{doc.document_type.replace('_', ' ')}</span>
                           <span>{doc.chunk_count} chunks</span>
                           <span>Uploaded {formatDate(doc.upload_date)}</span>
                         </div>
                       </div>
                     </div>
                     <div className="flex items-center space-x-2">
                       <div className="flex items-center space-x-1">
                         <CheckCircle className="w-4 h-4 text-green-500" />
                         <span className="text-xs text-green-600">Active</span>
                       </div>
                       <button
                         onClick={() => deleteDocument(doc.document_id)}
                         className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                         title="Delete document"
                       >
                         <Trash2 className="w-4 h-4" />
                       </button>
                     </div>
                   </div>
                 </div>
               ))}
             </div>
           )}
         </div>
       )}
     </div>
   </div>
 );
};

export default LegalRAGApp;