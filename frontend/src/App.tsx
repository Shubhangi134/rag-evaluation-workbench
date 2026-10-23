
import { useEffect, useRef, useState } from 'react'
import './App.css'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'



type RetrievalMethod = 'Keyword' | 'TF-IDF' | 'Semantic'

type ChatSource = {
  document_id: string
  chunk_index: number
  text: string
}

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
}

function App() {
  const [method, setMethod] = useState<RetrievalMethod>('Semantic')
  const [compareAll, setCompareAll] = useState(false)
  const [benchmarkResults, setBenchmarkResults] = useState<Record<
  string,
  { precision_at_k: number; recall_at_k: number }
  > | null>(null)
  const [question, setQuestion] = useState('')
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>([])
  const [uploadMessage, setUploadMessage] = useState('')
  const [documentCount, setDocumentCount] = useState(0)
  const [savedDocuments, setSavedDocuments] = useState<
    { filename: string; chunk_count: number }[]
  >([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isAsking, setIsAsking] = useState(false)
  const [askError, setAskError] = useState('')
  useEffect(() => {
  const loadSavedDocuments = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/documents')

      if (!response.ok) {
        throw new Error('Failed to load saved documents.')
      }

      const data = await response.json()

      setSavedDocuments(data.documents)
      setDocumentCount(data.documents.length)
    } catch (error) {
      console.error('Could not load saved documents:', error)
    }
  }

  loadSavedDocuments()
}, [])
  const handleAskQuestion = async () => {
    const currentQuestion = question.trim()

    if (!currentQuestion || isAsking) return

    setIsAsking(true)
    if (compareAll) {
  await handleCompareAll(currentQuestion)
}
    setQuestion('')
    setAskError('')

    try {
      const params = new URLSearchParams({
        query: currentQuestion,
        top_k: '3',
        history: JSON.stringify(conversationHistory),
        method,
      })

      const response = await fetch(
        `http://127.0.0.1:8000/ask?${params.toString()}`,
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get an answer.')
      }

      setConversationHistory((history) => [
        ...history,
        { role: 'user', content: currentQuestion },
        {
          role: 'assistant',
          content: data.answer,
          sources: data.retrieved_chunks ?? [],
        },
      ])
    } catch (error) {
      setAskError(
        error instanceof Error
          ? error.message
          : 'Could not connect to the backend.',
      )
      setQuestion(currentQuestion)
    } finally {
      setIsAsking(false)
    }
  }
const handleCompareAll = async (query: string) => {
  const currentQuestion = query.trim()

  if (!currentQuestion) {
    setAskError('Enter a question before comparing retrieval methods.')
    return
  }

  try {
    setAskError('')

    const params = new URLSearchParams({
      query: currentQuestion,
      top_k: '3',
    })

    const response = await fetch(
      `http://127.0.0.1:8000/benchmark?${params.toString()}`
    )

    const data = await response.json()

    if (!response.ok) {
      throw new Error(data.detail || 'Benchmark failed.')
    }

    setBenchmarkResults(data.results)
  } catch (error) {
    setAskError(
      error instanceof Error
        ? error.message
        : 'Could not run the benchmark.'
    )
  }
}
  const handleFileUpload = async (file: File) => {
    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
      setUploadMessage('Please select a PDF file.')
      return
    }

    setUploadMessage('Uploading PDF...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://127.0.0.1:8000/documents/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed.')
      }

      setDocumentCount((count) => count + 1)
      setUploadMessage(
        `${data.filename} uploaded successfully (${data.chunk_count} chunks).`,
      )
    } catch (error) {
      setUploadMessage(
        error instanceof Error
          ? error.message
          : 'Could not connect to the backend.',
      )
    }
  }

  const startNewQuestion = () => {
    setConversationHistory([])
    setQuestion('')
    setAskError('')
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <div>
            <h2>RAG Workbench</h2>
          </div>
        </div>

        <button className="new-chat" onClick={startNewQuestion}>
          ＋ New question
        </button>

        <p className="section-label">WORKSPACE</p>
        <button className="nav-item active">◈ Ask documents</button>
        <button className="nav-item">▦ Evaluate RAG</button>
        <button className="nav-item">⚙ Settings</button>

        <div className="sidebar-retrieval">
          <p className="section-label">RETRIEVAL METHOD</p>

          {(['Keyword', 'TF-IDF', 'Semantic'] as RetrievalMethod[]).map(
            (item) => (
              <button
                key={item}
                className={`nav-item ${method === item ? 'active' : ''}`}
                onClick={() => {
                  setMethod(item)
                  setCompareAll(false)
                }}
              >
                <span>
                  {item === 'Keyword' ? '⌕' : item === 'TF-IDF' ? '▥' : '✧'}
                </span>
                {item}
              </button>
            ),
          )}

          <label className="sidebar-compare">
            <input
              type="checkbox"
              checked={compareAll}
              onChange={(event) => {
                const checked = event.target.checked
                setCompareAll(checked)
                setBenchmarkResults(null)

                if (!checked) {
                  setBenchmarkResults(null)
                }
              }}
            />
            <span>Compare all 3</span>
          </label>
          {benchmarkResults && (
  <details className="benchmark-results">
    <summary>Retrieval Comparison Result</summary>

    {Object.entries(benchmarkResults).map(([name, metrics]) => (
      <div className="benchmark-result" key={name}>
        <span className="benchmark-method">{name}</span>
        <span>
          Precision@3: {metrics.precision_at_k.toFixed(2)}
        </span>
        <span>
          Recall@3: {metrics.recall_at_k.toFixed(2)}
        </span>
      </div>
    ))}
  </details>
)}

        </div>

        <div className="sidebar-section">
          <div className="section-heading">
            <p className="section-label">YOUR DOCUMENTS</p>
            <span className="count">{documentCount}</span>
          </div>

          {savedDocuments.length > 0 ? (
  <div className="saved-documents">
    {savedDocuments.map((document) => (
      <div className="saved-document" key={document.filename}>
        <span className="document-icon">▤</span>
        <div>
          <p>{document.filename}</p>
          <span>{document.chunk_count} chunks</span>
        </div>
      </div>
    ))}
  </div>
) : (
  <div className="empty-documents">
    <span className="document-icon">▤</span>
    <p>No documents yet</p>
    <span>Upload a PDF to get started.</span>
  </div>
)}

          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf,.pdf"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) {
                void handleFileUpload(file)
                event.target.value = ''
              }
            }}
          />

          <button
            className="upload-button"
            onClick={() => fileInputRef.current?.click()}
          >
            ＋ Upload PDF
          </button>

          {uploadMessage && (
            <p className="upload-message" role="status">
              {uploadMessage}
            </p>
          )}
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="breadcrumb">
              WORKSPACE / <span>ASK DOCUMENTS</span>
            </p>
            <h1>Ask your documents</h1>
          </div>
          <div className="topbar-right">
            <div className="avatar">S</div>
          </div>
        </header>

        <section className="workspace">
          <div className="workspace-intro">
            <div>
              <p>Get grounded answers with inspectable source passages.</p>
            </div>
            <span className="workspace-tag">RAG EVALUATION</span>
          </div>

          {conversationHistory.length > 0 && (
            <section className="conversation-history">
              <p className="eyebrow">CONVERSATION</p>

              {Array.from(
                { length: Math.ceil(conversationHistory.length / 2) },
                (_, index) => {
                  const userMessage = conversationHistory[index * 2]
                  const assistantMessage = conversationHistory[index * 2 + 1]

                  return (
                    <div className="chat-exchange" key={index}>
                      <div className="chat-question">
                        <span className="chat-role">You</span>
                        <p>{userMessage.content}</p>
                      </div>

                      {assistantMessage && (
                        <div className="chat-answer">
                          <span className="chat-role">Answer</span>
                          <div className="chat-answer-text">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {assistantMessage.content}
                          </ReactMarkdown>                            </div>

                          {assistantMessage.sources &&
                            assistantMessage.sources.length > 0 && (
                              <details className="chat-sources">
                                <summary>
                                  View sources ({assistantMessage.sources.length})
                                </summary>

                                {assistantMessage.sources.map((source) => (
                                  <details
                                    className="source-card"
                                    key={`${source.document_id}-${source.chunk_index}`}
                                  >
                                    <summary className="source-card-header">
                                      <strong>{source.document_id}</strong>
                                      <span>Chunk {source.chunk_index} ⌄</span>
                                    </summary>
                                    <p>{source.text}</p>
                                  </details>
                                ))}
                              </details>
                            )}
                        </div>
                      )}
                    </div>
                  )
                },
              )}
            </section>
          )}

          {askError && (
            <div className="answer-error" role="alert">
              <strong>Something went wrong</strong>
              <p>{askError}</p>
            </div>
          )}

          <section className="question-section">
            <div className="question-heading">
              <h3>Ask a question</h3>
              <span>YOUR DOCUMENTS, YOUR ANSWERS</span>
            </div>

            <div className="question-box">
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="e.g. What are the key concepts explained in this document?"
                rows={3}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    void handleAskQuestion()
                  }
                }}
              />

              <div className="question-actions">
                <span>↳ Answers will be grounded in retrieved passages.</span>
                <button
                  className="ask-button"
                  disabled={!question.trim() || isAsking}
                  onClick={() => void handleAskQuestion()}
                >
                  {isAsking ? 'Generating...' : 'Ask question'} <span>→</span>
                </button>
              </div>
            </div>
          </section>
        </section>
      </main>
    </div>
  )
}

export default App
