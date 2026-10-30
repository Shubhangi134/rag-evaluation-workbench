
import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'
const RETRIEVAL_METHOD = 'Semantic'

type ChatSource = {
  document_id: string
  chunk_index: number
  text: string
  page_number?: number
}

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  sources?: ChatSource[]
}

type SavedDocument = {
  filename: string
  chunk_count: number
}

const qasperResults = [
  {
    method: 'BM25',
    tokenF1: 0.1100,
    faithfulness: 0.9370,
    abstention: 46,
  },
  {
    method: 'TF-IDF',
    tokenF1: 0.1584,
    faithfulness: 0.8851,
    abstention: 42,
  },
  {
    method: 'Semantic',
    tokenF1: 0.1843,
    faithfulness: 0.8509,
    abstention: 28,
  },
]

const scifactResults = [
  {
    method: 'BM25',
    precision: 0.1567,
    recall: 0.7243,
    ndcg: 0.6431,
    mrr: 0.6247,
  },
  {
    method: 'TF-IDF',
    precision: 0.1413,
    recall: 0.6466,
    ndcg: 0.5547,
    mrr: 0.5354,
  },
  {
    method: 'Semantic',
    precision: 0.1640,
    recall: 0.7379,
    ndcg: 0.6293,
    mrr: 0.5997,
  },
]

function App() {
  const [question, setQuestion] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessionError, setSessionError] = useState('')
  const [conversationHistory, setConversationHistory] = useState<ChatMessage[]>([])
  const [savedDocuments, setSavedDocuments] = useState<SavedDocument[]>([])
  const [uploadMessage, setUploadMessage] = useState('')
  const [askError, setAskError] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const [showEvaluation, setShowEvaluation] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const conversationEndRef = useRef<HTMLDivElement>(null)
  
  const loadSavedDocuments = async () => {
    if (!sessionId) return

    try {
      const response = await fetch(`${API_URL}/documents`, {
        headers: {
          'X-Session-ID': sessionId,
        },
      })

      if (!response.ok) {
        throw new Error('Failed to load documents.')
      }

      const data = await response.json()
      setSavedDocuments(data.documents)
    } catch (error) {
      console.error('Could not load documents:', error)
      setUploadMessage('Could not refresh the document list.')
    }
  }


  useEffect(() => {
    let cancelled = false

    const initializeSession = async () => {
      try {
        // Automatically create a session when the app opens.
        const response = await fetch(`${API_URL}/sessions`, {
          method: 'POST',
        })

        if (!response.ok) {
          throw new Error('Could not start a chat session.')
        }

        const data = await response.json()
        const newSessionId: string = data.session_id

        if (cancelled) return

        setSessionId(newSessionId)
        setSessionError('')

        // Load documents for the new session.
        const documentsResponse = await fetch(
          `${API_URL}/documents`,
          {
            headers: {
              'X-Session-ID': newSessionId,
            },
          },
        )

        if (!documentsResponse.ok) {
          throw new Error('Could not load documents.')
        }

        const documentsData = await documentsResponse.json()

        if (!cancelled) {
          setSavedDocuments(documentsData.documents)
        }
      } catch (error) {
        if (!cancelled) {
          setSessionError(
            error instanceof Error
              ? error.message
              : 'Could not connect to the backend.',
          )
        }
      }
    }

    void initializeSession()

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    })
  }, [conversationHistory])
  const handleFileUpload = async (file: File) => {
    if (
      file.type !== 'application/pdf' &&
      !file.name.toLowerCase().endsWith('.pdf')
    ) {
      setUploadMessage('Please select a PDF file.')
      return
    }

    setUploadMessage('Uploading PDF...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      
      
      if (!sessionId) {
        throw new Error(
          sessionError ||
            'Chat is not connected yet. Please refresh the page and try again.'
        )
      }

      const response = await fetch(`${API_URL}/documents/upload`, {
        method: 'POST',
        headers: {
          'X-Session-ID': sessionId,
        },
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed.')
      }

      setUploadMessage(`${data.filename} uploaded successfully.`)

      // Refresh the list immediately after a successful upload.
      await loadSavedDocuments()
    } catch (error) {
      setUploadMessage(
        error instanceof Error
          ? error.message
          : 'Could not connect to the backend.',
      )
    }
  }

  const handleAskQuestion = async () => {
    const currentQuestion = question.trim()

    if (!currentQuestion || isAsking) return
    if (savedDocuments.length === 0) {
      setAskError(
    'Please upload a PDF document before asking a question.'
      )
  return
}

    setIsAsking(true)
    setQuestion('')
    setAskError('')

    try {
      const params = new URLSearchParams({
        query: currentQuestion,
        top_k: '3',
        method: RETRIEVAL_METHOD,
        history: JSON.stringify(conversationHistory),
      })

      if (!sessionId) {
  throw new Error(
    sessionError ||
      'Chat is not connected yet. Please refresh the page.'
  )
}

      const response = await fetch(
        `${API_URL}/ask?${params.toString()}`,
        {
          headers: {
            'X-Session-ID': sessionId,
          },
        },
      )
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get an answer.')
      }

      setConversationHistory((history) => [
        ...history,
        {
          role: 'user',
          content: currentQuestion,
        },
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

  const startNewChat = () => {
    setConversationHistory([])
    setQuestion('')
    setAskError('')
    setShowEvaluation(false)
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="new-chat" onClick={startNewChat}>
          ＋ New Chat
        </button>

        <div className="sidebar-section">
          <div className="section-heading">
            <p className="section-label">YOUR DOCUMENTS</p>
            <span className="count">{savedDocuments.length}</span>
          </div>

          {savedDocuments.length > 0 ? (
            <div className="saved-documents">
              {savedDocuments.map((document) => (
                <div
                  className="saved-document"
                  key={document.filename}
                >
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

        <button
          className={`nav-item evaluation-nav ${
            showEvaluation ? 'active' : ''
          }`}
          onClick={() => setShowEvaluation(true)}
          aria-expanded={showEvaluation}
          aria-controls="evaluation-panel"
        >
          <span>▦</span>
          Evaluation
        </button>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <h1>RAG Chat Bot</h1>
          <p className="topbar-subtitle">
            Chat with your PDFs
          </p>
        </header>

        <div className="app-content">
          <section className="workspace">

            {conversationHistory.length > 0 && (
              <section className="conversation-history">
                {Array.from(
                  {
                    length: Math.ceil(
                      conversationHistory.length / 2,
                    ),
                  },
                  (_, index) => {
                    const userMessage =
                      conversationHistory[index * 2]

                    const assistantMessage =
                      conversationHistory[index * 2 + 1]

                    return (
                      <div className="chat-exchange" key={index}>
                        <div className="chat-question">
                          <span className="chat-role">You</span>
                          <p>{userMessage.content}</p>
                        </div>

                        {assistantMessage && (
                          <div className="chat-answer">
                            <span className="chat-role">
                              Assistant
                            </span>

                            <div className="chat-answer-text">
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                              >
                                {assistantMessage.content}
                              </ReactMarkdown>
                            </div>

                            {assistantMessage.sources &&
                              assistantMessage.sources.length > 0 && (
                                <details className="chat-sources">
                                  <summary>
                                    View sources (
                                    {assistantMessage.sources.length}
                                    )
                                  </summary>

                                  {assistantMessage.sources.map(
                                    (source) => (
                                      <details
                                        className="source-card"
                                        key={`${source.document_id}-${source.chunk_index}`}
                                      >
                                        <summary className="source-card-header">
                                          <strong>
                                            {source.document_id}
                                          </strong>

                                          <span>
                                            {source.page_number != null
                                              ? `Page ${source.page_number}`
                                              : 'Source passage'}
                                          </span>
                                        </summary>

                                        <p>{source.text}</p>
                                      </details>
                                    ),
                                  )}
                                </details>
                              )}
                          </div>
                        )}
                      </div>
                    )
                  },
                )}
                <div ref={conversationEndRef} />
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
              </div>

              <div className="question-box">
                <textarea
                  value={question}
                  onChange={(event) =>
                    setQuestion(event.target.value)
                  }
                  placeholder="Ask anything about your uploaded documents..."
                  rows={3}
                  onKeyDown={(event) => {
                    if (
                      event.key === 'Enter' &&
                      !event.shiftKey
                    ) {
                      event.preventDefault()
                      void handleAskQuestion()
                    }
                  }}
                />

                <div className="question-actions">
                  <span>
                    Answers are grounded in uploaded PDFs.
                  </span>

                  <button
                    className="ask-button"
                    disabled={!question.trim() || isAsking}
                    onClick={() => void handleAskQuestion()}
                  >
                    {isAsking ? 'Generating...' : 'Send'}
                    <span>→</span>
                  </button>
                </div>
              </div>
            </section>
          </section>

          {showEvaluation && (
            <aside
              id="evaluation-panel"
              className="evaluation-panel"
              aria-label="Evaluation results"
            >
              <div className="evaluation-header">
                <div>
                  <h2>Evaluation</h2>
                  <p>Retrieval &amp; answer quality</p>
                </div>

                <button
                  className="evaluation-close"
                  onClick={() => setShowEvaluation(false)}
                  aria-label="Close evaluation panel"
                  title="Close evaluation"
                >
                  ×
                </button>
              </div>

              <div className="evaluation-content">
                <section className="evaluation-section">
                  <h3>QASPER Benchmark</h3>
                  <p>
                    50 questions · Top 3 retrieved passages
                  </p>

                  <div className="evaluation-table-wrapper">
                    <table className="evaluation-table">
                      <thead>
                        <tr>
                          <th>Method</th>
                          <th>Token F1</th>
                          <th>Faithfulness</th>
                          <th>Abstention</th>
                        </tr>
                      </thead>
                      <tbody>
                        {qasperResults.map((result) => (
                          <tr key={result.method}>
                            <td>{result.method}</td>
                            <td>{result.tokenF1.toFixed(4)}</td>
                            <td>
                              {result.faithfulness.toFixed(4)}
                            </td>
                            <td>{result.abstention}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <p className="evaluation-note">
                    Token F1 measures answer overlap with
                    reference answers. Faithfulness measures
                    whether generated claims are supported by
                    retrieved evidence.
                  </p>

                  <p className="evaluation-note">
                    <strong>Abstention rate:</strong> The
                    percentage of questions the chatbot chose
                    not to answer because it found insufficient
                    evidence.
                  </p>
                </section>

                <section className="evaluation-section">
                  <h3>SciFact Retrieval Benchmark</h3>
                  <p>
                    300 queries · 5,183 documents · Top 5
                    results
                  </p>

                  <div className="evaluation-table-wrapper">
                    <table className="evaluation-table">
                      <thead>
                        <tr>
                          <th>Method</th>
                          <th>P@5</th>
                          <th>R@5</th>
                          <th>nDCG@5</th>
                          <th>MRR@5</th>
                        </tr>
                      </thead>
                      <tbody>
                        {scifactResults.map((result) => (
                          <tr key={result.method}>
                            <td>{result.method}</td>
                            <td>{result.precision.toFixed(4)}</td>
                            <td>{result.recall.toFixed(4)}</td>
                            <td>{result.ndcg.toFixed(4)}</td>
                            <td>{result.mrr.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <p className="evaluation-note">
                    BM25 achieved the highest nDCG@5 and
                    MRR@5. Semantic retrieval achieved the
                    highest Precision@5 and Recall@5.
                  </p>
                </section>

                <p className="evaluation-disclaimer">
                  These are saved offline benchmark results,
                  not live scores for the uploaded PDFs.
                </p>
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
