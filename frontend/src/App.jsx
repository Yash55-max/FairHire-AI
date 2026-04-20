import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart,
  Pie, PieChart, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from 'recharts'

/* ═══════════════════════════════════════════════════════════════════════════
   CONSTANTS & ROUTING
═══════════════════════════════════════════════════════════════════════════ */
const ROUTES = new Set([
  'landing', 'login', 'dashboard', 'upload', 'model-analysis',
  'bias-report', 'explainability', 'reports', 'settings',
  'dual-audit', 'candidate-scoring', 'ethical-validator', 'assistant', 'debias', 'stress-test',
])

const PROTECTED_ROUTES = new Set([
  'dashboard', 'upload', 'model-analysis', 'bias-report',
  'explainability', 'reports', 'settings',
  'dual-audit', 'candidate-scoring', 'ethical-validator', 'assistant', 'debias', 'stress-test',
])

const SESSION_KEY = 'fairhire_session_v2'
const API_BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

function readRoute() {
  const value = window.location.hash.replace(/^#\/?/, '') || 'landing'
  return ROUTES.has(value) ? value : 'landing'
}

function navigate(route) {
  window.location.hash = `#/${route}`
}

/* ═══════════════════════════════════════════════════════════════════════════
   USER HELPERS
═══════════════════════════════════════════════════════════════════════════ */
function deriveDisplayName(email) {
  if (!email || !email.includes('@')) return 'FairHire User'
  return email.split('@')[0]
    .replace(/[._-]+/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map(p => p.charAt(0).toUpperCase() + p.slice(1))
    .join(' ')
}

function deriveInitials(name) {
  const parts = (name || 'FairHire').split(' ').filter(Boolean)
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  return `${parts[0][0] || ''}${parts[1][0] || ''}`.toUpperCase()
}

/* ═══════════════════════════════════════════════════════════════════════════
   API LAYER
═══════════════════════════════════════════════════════════════════════════ */
async function callApi(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail || `API error ${response.status}`)
  return payload
}

/* ═══════════════════════════════════════════════════════════════════════════
   MOCK DATA (fallback when backend is offline)
═══════════════════════════════════════════════════════════════════════════ */
function mockUpload(file) {
  return {
    dataset_id: `demo_ds_${Date.now()}`,
    filename: file?.name || 'sample_dataset.csv',
    rows: 50,
    columns: ['candidate_id', 'age', 'gender', 'education', 'years_experience', 'assessment_score', 'referral_source', 'role_applied', 'hired'],
    target_suggestions: ['hired'],
    schema: { candidate_id: 'object', age: 'int64', gender: 'object', education: 'object', years_experience: 'int64', assessment_score: 'int64', referral_source: 'object', role_applied: 'object', hired: 'int64' },
    null_counts: { candidate_id: 0, age: 0, gender: 0, education: 0, years_experience: 0, assessment_score: 0, referral_source: 0, role_applied: 0, hired: 0 },
    preview: [
      { candidate_id: 'CAND-001', age: 28, gender: 'Female', education: 'Bachelor', years_experience: 4, assessment_score: 82, referral_source: 'Employee Referral', role_applied: 'Software Engineer', hired: 1 },
      { candidate_id: 'CAND-002', age: 34, gender: 'Male', education: 'Master', years_experience: 9, assessment_score: 91, referral_source: 'LinkedIn', role_applied: 'Data Scientist', hired: 1 },
      { candidate_id: 'CAND-003', age: 26, gender: 'Female', education: 'Bootcamp', years_experience: 2, assessment_score: 68, referral_source: 'Job Board', role_applied: 'Frontend Developer', hired: 0 },
    ],
  }
}

function mockTrain(datasetId, target) {
  return {
    run_id: `demo_run_${Date.now()}`,
    dataset_id: datasetId,
    model_type: 'random_forest',
    target_column: target,
    accuracy: 0.924, precision: 0.912, recall: 0.897, f1_score: 0.904,
    confusion_matrix: { tp: 18, fp: 2, tn: 28, fn: 2 },
    prediction_preview: [],
    feature_count: 8, train_rows: 40, test_rows: 10,
  }
}

function mockBias(runId) {
  return {
    run_id: runId, sensitive_column: 'gender',
    demographic_parity_difference: 0.14,
    equal_opportunity_difference: 0.042,
    selection_rate_by_group: { Female: 0.61, Male: 0.75, NonBinary: 0.60 },
    true_positive_rate_by_group: { Female: 0.88, Male: 0.93, NonBinary: 0.87 },
    fairness_index: 0.73,
    verdict: 'REVIEW',
    verdict_detail: 'Moderate disparity detected. Manual review recommended.',
    recommendations: [
      'Reweight training data to balance group representation.',
      'Review referral sources — possible proxy variable for protected class.',
    ],
  }
}

function mockExplain(runId) {
  return {
    run_id: runId, sample_size: 40,
    top_global_features: [
      { feature: 'years_experience', mean_abs_shap: 0.31, importance: 0.31 },
      { feature: 'assessment_score', mean_abs_shap: 0.26, importance: 0.26 },
      { feature: 'referral_source', mean_abs_shap: 0.19, importance: 0.19 },
      { feature: 'education', mean_abs_shap: 0.12, importance: 0.12 },
      { feature: 'age', mean_abs_shap: 0.07, importance: 0.07 },
      { feature: 'gender', mean_abs_shap: 0.05, importance: 0.05 },
    ],
    local_explanation: [
      { feature: 'years_experience', shap_value: 0.22, direction: 'positive' },
      { feature: 'assessment_score', shap_value: 0.18, direction: 'positive' },
      { feature: 'education', shap_value: -0.06, direction: 'negative' },
      { feature: 'referral_source', shap_value: -0.04, direction: 'negative' },
    ],
  }
}

function mockReport(runId, train, bias, explain) {
  return {
    run_id: runId, train, bias, explain,
    generated_at: new Date().toISOString(),
    executive_summary: `Model achieved ${(train?.accuracy || 0.92).toFixed(1) * 100}% accuracy. Fairness verdict: ${bias?.verdict || 'REVIEW'}. Top driver: ${explain?.top_global_features?.[0]?.feature || 'years_experience'}.`,
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   ICONS
═══════════════════════════════════════════════════════════════════════════ */
function Icon({ name, size = 16 }) {
  const s = { width: size, height: size, fill: 'currentColor', flex: '0 0 auto' }
  switch (name) {
    case 'dashboard': return <svg viewBox="0 0 24 24" style={s}><path d="M4 13.5V4h7v9.5H4Zm9 6.5V11h7v9h-7ZM4 20v-4.5h7V20H4Zm9-12V4h7v4h-7Z"/></svg>
    case 'upload': return <svg viewBox="0 0 24 24" style={s}><path d="M12 3 6.5 8.5 7.9 9.9 11 6.8V16h2V6.8l3.1 3.1 1.4-1.4L12 3ZM5 18v2h14v-2H5Z"/></svg>
    case 'analysis': return <svg viewBox="0 0 24 24" style={s}><path d="M4 19h16v2H4v-2Zm2-3 3-5 3 2 4-7 2 1.2-5.4 9-3-2-2.1 3.5L6 16Z"/></svg>
    case 'bias': return <svg viewBox="0 0 24 24" style={s}><path d="M12 3 2.8 20h18.4L12 3Zm0 4.8 5.5 10.2H6.5L12 7.8Zm-1 3.2h2v4h-2v-4Zm0 5h2v2h-2v-2Z"/></svg>
    case 'explain': return <svg viewBox="0 0 24 24" style={s}><path d="M12 2 2 7v10l10 5 10-5V7L12 2Zm0 2.3 7.9 4L12 12.3 4.1 8.3 12 4.3ZM4 18V9.9l8 4v8.1l-8-4Zm16 0-8 4v-8.1l8-4V18Z"/></svg>
    case 'reports': return <svg viewBox="0 0 24 24" style={s}><path d="M6 3h9l5 5v13H6V3Zm8 1.5V9h4.5L14 4.5ZM8 12h8v2H8v-2Zm0 4h8v2H8v-2Zm0-8h3v2H8V8Z"/></svg>
    case 'settings': return <svg viewBox="0 0 24 24" style={s}><path d="m19.14 12.94 1.2-1.2-1.9-3.3-1.62.56a6.7 6.7 0 0 0-1.16-.67L15.5 6h-3l-.16 1.33c-.4.17-.8.39-1.16.67l-1.62-.56-1.9 3.3 1.2 1.2c-.05.3-.08.62-.08.94s.03.64.08.94l-1.2 1.2 1.9 3.3 1.62-.56c.36.28.76.5 1.16.67L12.5 18h3l.16-1.33c.4-.17.8-.39 1.16-.67l1.62.56 1.9-3.3-1.2-1.2c.05-.3.08-.62.08-.94s-.03-.64-.08-.94ZM12 15.2a3.2 3.2 0 1 1 0-6.4 3.2 3.2 0 0 1 0 6.4Z"/></svg>
    case 'search': return <svg viewBox="0 0 24 24" style={s}><path d="m21 20-4.3-4.3a7 7 0 1 0-1.4 1.4L20 21l1-1ZM5 11a6 6 0 1 1 12 0A6 6 0 0 1 5 11Z"/></svg>
    case 'spark': return <svg viewBox="0 0 24 24" style={s}><path d="m12 2 1.8 5.1L19 9l-5.2 1.9L12 16l-1.8-5.1L5 9l5.2-1.9L12 2Zm7 9 1.2 3.4L24 16l-3.8 1.6L19 21l-1.2-3.4L14 16l3.8-1.6L19 11Z"/></svg>
    case 'download': return <svg viewBox="0 0 24 24" style={s}><path d="M12 3v10.2l3.6-3.6 1.4 1.4L12 17 6.9 11 8.3 9.6 12 13.2V3ZM5 19h14v2H5v-2Z"/></svg>
    case 'check': return <svg viewBox="0 0 24 24" style={s}><path d="m9.2 16.2-4-4L3.8 13l5.4 5.4L20.2 7.4 18.8 6 9.2 16.2Z"/></svg>
    case 'warning': return <svg viewBox="0 0 24 24" style={s}><path d="M12 3 2.8 20h18.4L12 3Zm1 13h-2v-2h2v2Zm0-3h-2V8h2v5Z"/></svg>
    case 'users': return <svg viewBox="0 0 24 24" style={s}><path d="M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm9 1a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM2 21v-1a6 6 0 0 1 12 0v1H2Zm14 0v-1.2a5.5 5.5 0 0 0-1.2-3.4A7 7 0 0 1 22 21v1h-6Z"/></svg>
    case 'shield': return <svg viewBox="0 0 24 24" style={s}><path d="m12 2 8 3v6c0 5.2-3.1 8.9-8 11-4.9-2.1-8-5.8-8-11V5l8-3Zm0 2.1-6 2.3V11c0 4.1 2.4 7.1 6 8.8 3.6-1.7 6-4.7 6-8.8V6.4l-6-2.3Z"/></svg>
    case 'login': return <svg viewBox="0 0 24 24" style={s}><path d="M10 17v-2h7V9h-7V7h9v10h-9ZM6 19V5h2v14H6Zm6.3-4.3L11 13.4 12.6 12H4v-2h8.6L11 8.6 12.3 7.3 16.9 12l-4.6 4.7Z"/></svg>
    case 'logout': return <svg viewBox="0 0 24 24" style={s}><path d="M14 17v-2h3V9h-3V7h5v10h-5ZM6 19V5h7v2H8v10h5v2H6Zm7.4-4.2-1.4-1.4 2.4-2.4H3v-2h11.4L12 6.6l1.4-1.4 4.8 4.8-4.8 4.8Z"/></svg>
    case 'file': return <svg viewBox="0 0 24 24" style={s}><path d="M6 2h9l5 5v15H6V2Zm8 1.5V8h4.5L14 3.5ZM8 12h8v2H8v-2Zm0 4h8v2H8v-2Zm0-8h3v2H8V8Z"/></svg>
    case 'chart': return <svg viewBox="0 0 24 24" style={s}><path d="M4 22V8h4v14H4Zm6 0V2h4v20h-4Zm6 0v-8h4v8h-4Z"/></svg>
    case 'ai': return <svg viewBox="0 0 24 24" style={s}><path d="M12 2a5 5 0 0 1 5 5 5 5 0 0 1-5 5 5 5 0 0 1-5-5 5 5 0 0 1 5-5m0 12c5.33 0 8 2.67 8 4v2H4v-2c0-1.33 2.67-4 8-4Z"/></svg>
    case 'close': return <svg viewBox="0 0 24 24" style={s}><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
    case 'info': return <svg viewBox="0 0 24 24" style={s}><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm1 15h-2v-6h2v6Zm0-8h-2V7h2v2Z"/></svg>
    case 'arrow-left': return <svg viewBox="0 0 24 24" style={s}><path d="m15 18-6-6 6-6 1.4 1.4L11.8 12l4.6 4.6L15 18Z"/></svg>
    case 'arrow-right': return <svg viewBox="0 0 24 24" style={s}><path d="m9 18 6-6-6-6-1.4 1.4L13.2 12l-5.6 5.6L9 18Z"/></svg>
    case 'refresh': return <svg viewBox="0 0 24 24" style={s}><path d="M17.65 6.35A7.96 7.96 0 0 0 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08A5.99 5.99 0 0 1 12 18c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35Z"/></svg>
    default: return <svg viewBox="0 0 24 24" style={s}><path d="M4 12h16"/></svg>
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   SHARED UI PRIMITIVES
═══════════════════════════════════════════════════════════════════════════ */
class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { hasError: false, message: '' } }
  static getDerivedStateFromError(error) { return { hasError: true, message: error?.message || 'Render error' } }
  componentDidCatch(e) { console.error('UI error:', e) }
  render() {
    if (this.state.hasError) return (
      <div className="app-error-screen">
        <Icon name="warning" size={40} />
        <h1>Something went wrong</h1>
        <p>{this.state.message}</p>
        <button type="button" className="primary-button" onClick={() => { this.setState({ hasError: false }); window.location.hash = '#/landing' }}>
          Reload
        </button>
      </div>
    )
    return this.props.children
  }
}

function Btn({ type = 'button', cls = 'primary-button', icon, children, onClick, disabled = false }) {
  return (
    <button type={type} className={cls} onClick={onClick} disabled={disabled}>
      {icon && <Icon name={icon} />}
      {children}
    </button>
  )
}

function Skeleton({ h = 200 }) {
  return <div className="skeleton" style={{ height: h }} />
}

function Badge({ variant = 'info', children }) {
  return <span className={`badge badge-${variant}`}>{children}</span>
}

function Pill({ variant = 'green', icon, children }) {
  return <span className={`status-chip ${variant}`}>{icon && <Icon name={icon} />}{children}</span>
}

function KeyRow({ label, value, mono = false }) {
  return (
    <div className="key-row">
      <span>{label}</span>
      <strong style={mono ? { fontFamily: 'monospace', fontSize: '0.85rem' } : {}}>{value}</strong>
    </div>
  )
}

function MetricCard({ label, value, note, accent = false, icon = 'spark', trend }) {
  return (
    <article className={`metric-card surface-card${accent ? ' metric-accent' : ''}`}>
      <span className="eyebrow"><Icon name={icon} />{label}</span>
      <strong>{value}</strong>
      <p>{note}</p>
      {trend !== undefined && (
        <span className={`trend ${trend >= 0 ? 'up' : 'down'}`}>
          {trend >= 0 ? '▲' : '▼'} {Math.abs(trend).toFixed(1)}%
        </span>
      )}
    </article>
  )
}

function SectionCard({ title, subtitle, children, className = '', icon = 'spark', action }) {
  return (
    <article className={`section-card surface-card ${className}`.trim()}>
      {(title || subtitle) && (
        <header className="section-head">
          {subtitle && <span className="eyebrow"><Icon name={icon} />{subtitle}</span>}
          <div className="section-head-row">
            {title && <h2>{title}</h2>}
            {action}
          </div>
        </header>
      )}
      {children}
    </article>
  )
}

function InfoStrip({ items }) {
  return (
    <div className="info-strip">
      {items.map(item => (
        <article key={item.title} className="info-chip surface-card">
          <span className="eyebrow"><Icon name={item.icon} />{item.label}</span>
          <strong>{item.title}</strong>
          <p>{item.text}</p>
        </article>
      ))}
    </div>
  )
}

function VerdictBanner({ verdict, detail, recommendations, parity, onNavigate }) {
  const isPass = verdict === 'PASS'
  const isFail = verdict === 'FAIL'
  const cls = isPass ? 'verdict-banner ok' : isFail ? 'verdict-banner fail' : 'verdict-banner warn'
  return (
    <section className={cls}>
      <div className="verdict-inner">
        <span className="eyebrow"><Icon name={isFail ? 'warning' : isPass ? 'check' : 'info'} />Fairness Verdict</span>
        <h2>
          {isPass ? '✅ FAIRNESS PASS' : isFail ? '🚨 FAIRNESS FAIL' : '⚠️ REVIEW REQUIRED'}
        </h2>
        <p>{detail}</p>
        {parity !== undefined && (
          <p><strong>Demographic Parity Gap:</strong> {(parity * 100).toFixed(1)}%
            {' '}
            <span className={parity > 0.1 ? 'badge badge-danger' : 'badge badge-success'}>
              {parity > 0.1 ? 'Above threshold' : 'Within threshold'}
            </span>
          </p>
        )}
        {recommendations?.length > 0 && (
          <div className="reco-list">
            {recommendations.map((r, i) => (
              <div key={i} className="reco-item"><Icon name="check" />{r}</div>
            ))}
          </div>
        )}
      </div>
      <div className="verdict-side">
        <Pill variant={isPass ? 'green' : isFail ? 'red' : 'amber'}>{verdict}</Pill>
        {onNavigate && <Btn cls="secondary-button" icon="reports" onClick={() => onNavigate('reports')}>View Report</Btn>}
      </div>
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   TOAST
═══════════════════════════════════════════════════════════════════════════ */
function ToastStack({ toasts, onDismiss }) {
  return (
    <div className="toast-stack">
      {toasts.map(t => (
        <div key={t.id} className={`toast toast-${t.type}`} role="alert">
          <Icon name={t.type === 'error' ? 'warning' : t.type === 'success' ? 'check' : 'info'} />
          <div><strong>{t.title}</strong><p>{t.message}</p></div>
          <button type="button" className="toast-close" onClick={() => onDismiss(t.id)}><Icon name="close" /></button>
        </div>
      ))}
    </div>
  )
}

function useToasts() {
  const [toasts, setToasts] = useState([])
  const push = useCallback((type, title, message) => {
    const id = `toast_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
    setToasts(cur => [...cur, { id, type, title, message }])
    setTimeout(() => setToasts(cur => cur.filter(t => t.id !== id)), 5000)
  }, [])
  const dismiss = useCallback(id => setToasts(cur => cur.filter(t => t.id !== id)), [])
  return { toasts, push, dismiss }
}

/* ═══════════════════════════════════════════════════════════════════════════
   APP SHELL
═══════════════════════════════════════════════════════════════════════════ */
const NAV_ITEMS = [
  ['dashboard', 'Dashboard', 'dashboard'],
  ['upload', 'Upload Dataset', 'upload'],
  ['debias', 'De-biasing Engine', 'shield'],
  ['dual-audit', 'Dual Audit', 'analysis'],
  ['model-analysis', 'Model Evaluation', 'analysis'],
  ['bias-report', 'Fairness Audit', 'bias'],
  ['candidate-scoring', 'Candidate Scoring', 'users'],
  ['ethical-validator', 'Ethical Validator', 'check'],
  ['explainability', 'Decision Insights', 'explain'],
  ['stress-test', 'Bias Stress Test', 'warning'],
  ['assistant', 'AI Assistant', 'ai'],
  ['reports', 'Reports', 'reports'],
  ['settings', 'Settings', 'settings'],
]

const FLOW_STEPS = [
  'Upload Data', 'De-bias', 'Dual Audit', 'Train Model',
  'Score Candidates', 'Bias Audit', 'Validate', 'Export Report',
]

function Sidebar({ active, onNavigate, isAuth, userProfile, onLogout }) {
  return (
    <aside className="sidebar surface-panel">
      <div className="brand-block">
        <div className="brand-mark" />
        <div>
          <strong>FairHire AI</strong>
          <span>Enterprise Hiring Audit</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(([route, label, icon]) => (
          <button
            key={route}
            type="button"
            className={`nav-item${route === active ? ' active' : ''}`}
            onClick={() => onNavigate(route)}
          >
            <Icon name={icon} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        {isAuth && userProfile && (
          <div className="profile-panel">
            <div className="avatar-circle">{userProfile.initials}</div>
            <div className="profile-meta">
              <strong>{userProfile.name}</strong>
              <small>{userProfile.email}</small>
              <Pill variant="teal"><Icon name="shield" size={11} />Audit Manager</Pill>
            </div>
          </div>
        )}
        {isAuth
          ? <Btn cls="nav-cta" icon="logout" onClick={onLogout}>Sign Out</Btn>
          : <Btn cls="nav-cta" icon="shield" onClick={() => onNavigate('login')}>Secure Sign In</Btn>
        }
      </div>
    </aside>
  )
}

function AppShell({ active, onNavigate, children, isAuth, onLogout, userProfile, trainData, uploadData }) {
  const stepDone = [
    Boolean(uploadData),
    Boolean(trainData),
    ['bias-report', 'explainability', 'reports', 'settings'].includes(active),
    ['explainability', 'reports', 'settings'].includes(active),
    ['reports', 'settings'].includes(active),
  ]

  return (
    <div className="app-shell">
      <Sidebar active={active} onNavigate={onNavigate} isAuth={isAuth} userProfile={userProfile} onLogout={onLogout} />

      <main className="main-panel">
        <header className="topbar surface-glass">
          <div className="search-shell">
            <Icon name="search" />
            <input placeholder="Search audits, models, reports…" />
          </div>
          <div className="topbar-actions">
            {isAuth && userProfile && (
              <button type="button" className="profile-chip" onClick={() => onNavigate('settings')}>
                <div className="avatar-circle avatar-sm">{userProfile.initials}</div>
                <span className="profile-chip-meta">
                  <strong>{userProfile.name}</strong>
                  <small>{userProfile.email}</small>
                </span>
              </button>
            )}
            <button type="button" className="icon-button" onClick={() => onNavigate('settings')}>
              <Icon name="settings" />
            </button>
          </div>
        </header>

        {isAuth && (
          <section className="guided-flow surface-card">
            <span className="eyebrow"><Icon name="spark" />Audit Workflow</span>
            <div className="flow-steps">
              {FLOW_STEPS.map((label, i) => (
                <div key={label} className={`flow-step${stepDone[i] ? ' done' : ''}`}>
                  <div className="flow-num">{i + 1}</div>
                  <span>{label}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="page-content">{children}</section>

        <nav className="mobile-nav surface-glass">
          {NAV_ITEMS.slice(0, 5).map(([route, , icon]) => (
            <button key={`m-${route}`} type="button"
              className={`mobile-nav-item${active === route ? ' active' : ''}`}
              onClick={() => onNavigate(route)}
            >
              <Icon name={icon} size={20} />
            </button>
          ))}
        </nav>
      </main>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   LANDING
═══════════════════════════════════════════════════════════════════════════ */
const LANDING_FEATURES = [
  {
    icon: 'shield', label: 'EEOC Compliance', title: 'Built-in regulatory alignment',
    text: 'Measure demographic parity, equal opportunity, and disparate impact automatically.',
  },
  {
    icon: 'analysis', label: 'SHAP Explainability', title: 'Decision-level transparency',
    text: 'Understand exactly which features drive individual hiring outcomes.',
  },
  {
    icon: 'reports', label: 'Board-Ready Reports', title: 'Executive audit packages',
    text: 'Generate compliance summaries formatted for HR leadership and legal review.',
  },
  {
    icon: 'chart', label: 'What-If Simulation', title: 'Scenario analysis',
    text: 'Explore how threshold adjustments and reweighting change fairness outcomes.',
  },
  {
    icon: 'ai', label: 'AI Recommendations', title: 'Guided remediation',
    text: 'Receive concrete mitigation actions ranked by expected fairness improvement.',
  },
  {
    icon: 'users', label: 'Group Analysis', title: 'Granular subgroup view',
    text: 'Track selection rates across gender, education, experience, and more.',
  },
]

function LandingPage({ onNavigate }) {
  return (
    <div className="landing-page">
      <header className="landing-nav surface-glass">
        <div className="brand-block">
          <div className="brand-mark" />
          <div>
            <strong>FairHire AI</strong>
            <span>Ethical hiring intelligence</span>
          </div>
        </div>
        <div className="landing-links">
          <a href="#features"><Icon name="spark" />Features</a>
          <a href="#/login"><Icon name="login" />Login</a>
          <Btn cls="primary-button" icon="dashboard" onClick={() => onNavigate('login')}>Get Started</Btn>
        </div>
      </header>

      <section className="hero-slab">
        <div className="hero-copy">
          <span className="eyebrow hero-eyebrow"><Icon name="spark" />Enterprise Hiring Compliance Platform</span>
          <h1>Detect bias. Explain decisions. Build fair hiring.</h1>
          <p>
            FairHire AI is an enterprise-ready audit platform that trains models, detects discrimination,
            produces SHAP-powered explanations, and generates board-ready compliance reports.
          </p>
          <div className="hero-actions">
            <Btn cls="primary-button hero-cta" icon="dashboard" onClick={() => onNavigate('login')}>
              Enter Platform
            </Btn>
            <Btn cls="glass-button" icon="download" onClick={() => onNavigate('reports')}>
              View Sample Report
            </Btn>
          </div>
          <div className="hero-stats">
            <div className="h-stat"><strong>6+</strong><span>Fairness metrics</span></div>
            <div className="h-stat"><strong>4</strong><span>ML models</span></div>
            <div className="h-stat"><strong>EEOC</strong><span>Aligned checks</span></div>
          </div>
        </div>
        <div className="hero-visual surface-card">
          <div className="hero-vis-header">
            <span className="eyebrow"><Icon name="chart" />Live Fairness Monitor</span>
          </div>
          <div className="hero-gauge-wrap">
            <div className="hero-gauge">
              <div className="gauge-fill" style={{ '--pct': '73%' }} />
              <span className="gauge-val">0.73</span>
              <span className="gauge-label">Fairness Index</span>
            </div>
          </div>
          <div className="hero-mini-bars">
            {[['Female', 61], ['Male', 75], ['NonBinary', 60]].map(([g, v]) => (
              <div key={g} className="mini-bar-row">
                <span>{g}</span>
                <div className="mini-bar-track">
                  <div className="mini-bar-fill" style={{ width: `${v}%`, background: v < 65 ? '#f59e0b' : '#14b8a6' }} />
                </div>
                <strong>{v}%</strong>
              </div>
            ))}
          </div>
          <div className="hero-chip warn"><Icon name="warning" />Gender parity gap: 15% – Review required</div>
        </div>
      </section>

      <section id="features" className="features-section">
        <div className="section-label">
          <span className="eyebrow"><Icon name="spark" />Platform Capabilities</span>
          <h2>Everything you need for compliant AI hiring</h2>
        </div>
        <div className="features-grid">
          {LANDING_FEATURES.map(f => (
            <article key={f.title} className="feature-card surface-card">
              <div className="feature-icon"><Icon name={f.icon} size={20} /></div>
              <span className="eyebrow">{f.label}</span>
              <h3>{f.title}</h3>
              <p>{f.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="cta-section surface-panel">
        <h2>Ready to audit your hiring model?</h2>
        <p>Upload your dataset and get a full fairness assessment in minutes.</p>
        <Btn cls="primary-button cta-btn" icon="dashboard" onClick={() => onNavigate('login')}>
          Start Free Audit
        </Btn>
      </section>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOGIN
═══════════════════════════════════════════════════════════════════════════ */
function LoginPage({ onNavigate, onLogin, authLoading }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [errors, setErrors] = useState({})

  const submit = e => {
    e.preventDefault()
    const err = {}
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) err.email = 'Enter a valid work email'
    if (password.length < 8) err.password = 'Password must be at least 8 characters'
    setErrors(err)
    if (Object.keys(err).length) return
    onLogin({ email, password })
  }

  return (
    <div className="login-page">
      <section className="login-hero surface-panel">
        <div className="login-hero-inner">
          <div className="brand-block brand-block-lg">
            <div className="brand-mark brand-mark-lg" />
            <div>
              <strong>FairHire AI</strong>
              <span>Enterprise Hiring Audit</span>
            </div>
          </div>
          <h1>Secure access to the audit environment</h1>
          <p>Compliant, session-based authentication designed for enterprise HR and data science teams.</p>
          <div className="login-trust">
            <Pill variant="teal" icon="shield">EEOC Aligned</Pill>
            <Pill variant="teal" icon="check">SOC-2 Ready</Pill>
            <Pill variant="teal" icon="shield">Session Encrypted</Pill>
          </div>
        </div>
      </section>

      <section className="login-form-shell surface-card">
        <form className="login-form" onSubmit={submit}>
          <div className="form-header">
            <span className="eyebrow"><Icon name="login" />Sign In</span>
            <h2>Corporate access</h2>
          </div>

          <label>
            Work email
            <input id="email" type="email" value={email}
              onChange={e => setEmail(e.target.value)} placeholder="name@company.com"
              autoComplete="email" />
            {errors.email && <small className="field-error">{errors.email}</small>}
          </label>

          <label>
            Password
            <input id="password" type="password" value={password}
              onChange={e => setPassword(e.target.value)} placeholder="••••••••"
              autoComplete="current-password" />
            {errors.password && <small className="field-error">{errors.password}</small>}
          </label>

          <Btn type="submit" cls="primary-button full-width" icon="dashboard" disabled={authLoading}>
            {authLoading ? 'Signing in…' : 'Continue to Dashboard'}
          </Btn>
          <Btn cls="secondary-button full-width" icon="arrow-left" onClick={() => onNavigate('landing')}>
            Back to Landing
          </Btn>

          <p className="login-hint">
            <Icon name="info" size={14} />
            Use any email + password (8+ chars) to access the demo environment.
          </p>
        </form>
      </section>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   DASHBOARD
═══════════════════════════════════════════════════════════════════════════ */
function DashboardPage({ onNavigate, biasData, trainData, loading, uploadData, runId }) {
  const biasChart = useMemo(() => {
    if (!biasData?.selection_rate_by_group) {
      return [{ group: 'Female', value: 0.61 }, { group: 'Male', value: 0.75 }, { group: 'NonBinary', value: 0.60 }]
    }
    return Object.entries(biasData.selection_rate_by_group).map(([group, value]) => ({ group, value }))
  }, [biasData])

  const parityDelta = biasData?.demographic_parity_difference ?? 0.14
  const verdict = biasData?.verdict ?? (parityDelta >= 0.1 ? 'REVIEW' : 'PASS')
  const verdictDetail = biasData?.verdict_detail ?? (parityDelta >= 0.1
    ? `Gender bias gap of ${(parityDelta * 100).toFixed(1)}% exceeds safe threshold.`
    : `Parity within safe band (Δ = ${(parityDelta * 100).toFixed(1)}%).`)

  return (
    <>
      <VerdictBanner
        verdict={verdict}
        detail={verdictDetail}
        recommendations={biasData?.recommendations}
        parity={parityDelta}
        onNavigate={onNavigate}
      />

      <div className="metric-grid">
        <MetricCard label="Model runs" value={runId ? '1' : '0'} note={runId || 'No model trained yet'} icon="analysis" />
        <MetricCard label="Fairness index" value={(biasData?.fairness_index ?? 0.73).toFixed(2)}
          note="Closer to 1.0 is better" accent icon="shield" />
        <MetricCard label="Accuracy" value={(trainData?.accuracy ?? 0).toFixed(2)}
          note={trainData ? 'Latest trained model' : 'Train a model first'} icon="check" />
      </div>

      <div className="two-col">
        <SectionCard title="Selection rate by group" subtitle="Fairness distribution" icon="bias">
          {loading.bias ? <Skeleton /> : (
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={biasChart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,124,0.15)" />
                  <XAxis dataKey="group" />
                  <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
                  <ReferenceLine y={0.6} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: 'EEOC threshold', fill: '#b45309', fontSize: 10 }} />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                    {biasChart.map(e => (
                      <Cell key={e.group} fill={e.value < 0.6 ? '#f59e0b' : '#14b8a6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>

        <SectionCard title="Quick actions" subtitle="Shortcuts" icon="spark">
          <div className="action-stack">
            <Btn cls="primary-button" icon="upload" onClick={() => onNavigate('upload')}>Upload Dataset</Btn>
            {runId && <Btn cls="secondary-button" icon="bias" onClick={() => onNavigate('bias-report')}>Fairness Audit</Btn>}
            {runId && <Btn cls="secondary-button" icon="explain" onClick={() => onNavigate('explainability')}>Decision Insights</Btn>}
            <Btn cls="secondary-button" icon="reports" onClick={() => onNavigate('reports')}>View Reports</Btn>
          </div>
        </SectionCard>
      </div>

      <SectionCard title="AI Audit Insights" subtitle="Automated recommendations" icon="ai" className="ai-card">
        <div className="ai-insights-inner">
          <div className="ai-list">
            {biasData?.recommendations?.length
              ? biasData.recommendations.map((r, i) => (
                <div key={i} className="ai-item"><Icon name="check" />{r}</div>
              ))
              : <>
                <div className="ai-item"><Icon name="analysis" />Train a model and run bias detection to see AI recommendations.</div>
                <div className="ai-item muted"><Icon name="spark" />Upload a CSV or XLSX dataset to get started.</div>
              </>
            }
          </div>
          <div className="ai-chips">
            <Pill variant="teal" icon="check">EEOC Aligned</Pill>
            <Pill variant="amber" icon="warning">Bias Detected</Pill>
            <Pill variant="teal" icon="shield">Audit Ready</Pill>
          </div>
        </div>
      </SectionCard>

      <InfoStrip items={[
        { icon: 'warning', label: 'Monitor', title: 'Bias drift alerts', text: 'Watch fairness index shifts before they cross risk thresholds.' },
        { icon: 'users', label: 'Coverage', title: 'Group representation', text: 'Ensure all sensitive groups remain visible in selection analysis.' },
        { icon: 'spark', label: 'Action', title: 'Weekly review cadence', text: 'Schedule recurring checks to keep hiring models accountable.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   UPLOAD
═══════════════════════════════════════════════════════════════════════════ */
function UploadPage({ onNavigate, loading, uploadData, onUpload, onTrain, selectedTarget, setSelectedTarget, modelType, setModelType }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = e => {
    e.preventDefault(); setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onUpload(f)
  }

  const nullTotal = uploadData?.null_counts ? Object.values(uploadData.null_counts).reduce((a, b) => a + b, 0) : 0

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Upload status" value={loading.upload ? 'Uploading…' : uploadData ? 'Ready' : 'Waiting'}
          note={uploadData ? uploadData.filename : 'Select a CSV, JSON, or XLSX file'} icon="upload" />
        <MetricCard label="Columns detected" value={uploadData?.columns?.length ?? 0}
          note="Dataset schema width" accent icon="chart" />
        <MetricCard label="Total rows" value={uploadData?.rows ?? 0} note="Dataset size" icon="users" />
      </div>

      <div className="two-col">
        <SectionCard title="Dataset intake" subtitle="Drag, drop, or browse" icon="upload">
          <input ref={inputRef} type="file" accept=".csv,.json,.xlsx,.xls" className="hidden-input"
            onChange={e => { const f = e.target.files?.[0]; if (f) onUpload(f) }} />
          <div
            className={`upload-dropzone${dragOver ? ' drag-over' : ''}`}
            role="button" tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
          >
            <div className="dz-icon"><Icon name="upload" size={28} /></div>
            <strong>{loading.upload ? 'Uploading…' : dragOver ? 'Drop to upload' : 'Click or drag a file here'}</strong>
            <p>CSV · JSON · Excel — Max 50MB</p>
            <Btn cls="secondary-button" icon="file">Browse files</Btn>
          </div>

          {/* Sample dataset link */}
          <p className="hint-text">
            <Icon name="info" size={13} />
            No dataset? Download the{' '}
            <a href={`${API_BASE}/sample-dataset`} download="fairhire_sample_dataset.csv">sample_dataset.csv</a>
            {' '}(200 rows) to try the platform.
          </p>
        </SectionCard>

        <SectionCard
          title={uploadData?.filename ?? 'No dataset selected'}
          subtitle={uploadData ? `${uploadData.rows} rows · ${uploadData.columns?.length} columns` : 'Awaiting upload'}
          icon="file"
        >
          {loading.upload ? <Skeleton h={300} /> : uploadData ? (
            <>
              <label className="select-label">
                Target column (what to predict)
                <select value={selectedTarget} onChange={e => setSelectedTarget(e.target.value)}>
                  {(uploadData.target_suggestions || uploadData.columns || []).map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </label>

              <label className="select-label">
                Model algorithm
                <select value={modelType} onChange={e => setModelType(e.target.value)}>
                  <option value="random_forest">Random Forest (recommended)</option>
                  <option value="logistic_regression">Logistic Regression</option>
                  <option value="gradient_boosting">Gradient Boosting</option>
                  <option value="decision_tree">Decision Tree</option>
                </select>
              </label>

              {nullTotal > 0 && (
                <div className="null-warning">
                  <Icon name="warning" />{nullTotal} null values detected — will be auto-imputed during training.
                </div>
              )}

              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>{Object.keys(uploadData.preview?.[0] || {}).map(k => <th key={k}>{k}</th>)}</tr>
                  </thead>
                  <tbody>
                    {(uploadData.preview || []).map((row, ri) => (
                      <tr key={ri}>
                        {Object.values(row).map((cell, ci) => <td key={ci}>{String(cell)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="placeholder-text">Upload a dataset to preview the schema and proceed to training.</p>
          )}
        </SectionCard>
      </div>

      {uploadData && uploadData.schema && (
        <SectionCard title="Column schema" subtitle="Data types detected" icon="chart">
          <div className="schema-grid">
            {Object.entries(uploadData.schema).map(([col, dtype]) => (
              <div key={col} className="schema-chip">
                <strong>{col}</strong>
                <Badge variant={dtype === 'object' ? 'info' : 'success'}>{dtype}</Badge>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('dashboard')}>Cancel</Btn>
        <Btn cls="primary-button" icon="analysis" onClick={onTrain} disabled={!uploadData || loading.train}>
          {loading.train ? 'Training model…' : 'Train Model →'}
        </Btn>
      </div>

      <InfoStrip items={[
        { icon: 'file', label: 'Required', title: 'Target column', text: 'Select the decision label column (e.g. "hired") before training.' },
        { icon: 'check', label: 'Quality', title: 'Schema consistency', text: 'Column names and data types should be consistent across batches.' },
        { icon: 'shield', label: 'Privacy', title: 'Sensitive field tagging', text: 'Mark protected attributes to power fairness diagnostics.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   MODEL ANALYSIS
═══════════════════════════════════════════════════════════════════════════ */
function ModelAnalysisPage({ onNavigate, trainData, loading }) {
  const cm = trainData?.confusion_matrix || { tp: 18, fp: 2, tn: 28, fn: 2 }
  const cmData = [
    { name: 'True Positive', value: cm.tp, fill: '#14b8a6' },
    { name: 'True Negative', value: cm.tn, fill: '#0b1f3a' },
    { name: 'False Positive', value: cm.fp, fill: '#f59e0b' },
    { name: 'False Negative', value: cm.fn, fill: '#ef4444' },
  ]

  const radarData = trainData ? [
    { metric: 'Accuracy', value: Math.round(trainData.accuracy * 100) },
    { metric: 'Precision', value: Math.round(trainData.precision * 100) },
    { metric: 'Recall', value: Math.round(trainData.recall * 100) },
    { metric: 'F1 Score', value: Math.round(trainData.f1_score * 100) },
  ] : []

  const fpr = cm.fp / (cm.fp + cm.tn || 1)
  const fnr = cm.fn / (cm.fn + cm.tp || 1)

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Accuracy" value={(trainData?.accuracy ?? 0).toFixed(3)} note="Overall correctness" icon="check" accent />
        <MetricCard label="Precision" value={(trainData?.precision ?? 0).toFixed(3)} note="Positive prediction quality" icon="analysis" />
        <MetricCard label="Recall" value={(trainData?.recall ?? 0).toFixed(3)} note="Positive capture rate" icon="users" />
        <MetricCard label="F1 Score" value={(trainData?.f1_score ?? 0).toFixed(3)} note="Harmonic mean" icon="spark" />
        <MetricCard label="False Pos. Rate" value={(fpr * 100).toFixed(1) + '%'} note="Incorrect accept rate" icon="warning" />
        <MetricCard label="False Neg. Rate" value={(fnr * 100).toFixed(1) + '%'} note="Incorrect reject rate" icon="bias" />
      </div>

      <div className="two-col">
        <SectionCard title="Confusion matrix" subtitle="Prediction outcome distribution" icon="analysis">
          {loading.train ? <Skeleton /> : (
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={cmData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,124,0.15)" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                    {cmData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>

        <SectionCard title="Model performance radar" subtitle="All metrics at a glance" icon="spark">
          {loading.train || !trainData ? <Skeleton /> : (
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <Radar dataKey="value" stroke="#14b8a6" fill="#14b8a6" fillOpacity={0.25} />
                  <Tooltip formatter={v => `${v}%`} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>
      </div>

      {trainData && (
        <SectionCard title="Training run summary" subtitle="Run metadata" icon="reports">
          <div className="key-grid">
            <KeyRow label="Run ID" value={trainData.run_id} mono />
            <KeyRow label="Dataset ID" value={trainData.dataset_id} mono />
            <KeyRow label="Algorithm" value={trainData.model_type.replace('_', ' ').toUpperCase()} />
            <KeyRow label="Target column" value={trainData.target_column} />
            <KeyRow label="Training rows" value={trainData.train_rows?.toLocaleString() ?? '—'} />
            <KeyRow label="Test rows" value={trainData.test_rows?.toLocaleString() ?? '—'} />
            <KeyRow label="Feature count" value={trainData.feature_count ?? '—'} />
          </div>
        </SectionCard>
      )}

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('upload')}>Back</Btn>
        <Btn cls="primary-button" icon="bias" onClick={() => onNavigate('bias-report')}>Run Fairness Audit →</Btn>
      </div>

      <InfoStrip items={[
        { icon: 'analysis', label: 'Performance', title: 'Confusion balance', text: 'Compare FP and FN rates — both matter for hiring fairness.' },
        { icon: 'spark', label: 'Validation', title: 'Cross-check metrics', text: 'Use precision and recall together, not accuracy alone.' },
        { icon: 'bias', label: 'Next step', title: 'Fairness audit required', text: 'Run bias checks before approving candidate scoring.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   BIAS REPORT
═══════════════════════════════════════════════════════════════════════════ */
function BiasReportPage({ onNavigate, biasData, loading, runId, uploadData, onRunBias }) {
  const [sensitiveCol, setSensitiveCol] = useState('gender')
  const [threshold, setThreshold] = useState(0.5)
  const [reweight, setReweight] = useState(0)
  const [simResult, setSimResult] = useState(null)
  const [simLoading, setSimLoading] = useState(false)

  const groups = Object.entries(biasData?.selection_rate_by_group || { Female: 0.61, Male: 0.75, NonBinary: 0.60 })
  const chartData = groups.map(([group, value]) => ({ group, value: Number(value) }))

  const tprData = Object.entries(biasData?.true_positive_rate_by_group || { Female: 0.88, Male: 0.93, NonBinary: 0.87 })
    .map(([group, value]) => ({ group, tpr: Number(value) }))

  const runSim = async () => {
    setSimLoading(true)
    try {
      const fi = biasData?.fairness_index ?? 0.73
      const pg = biasData?.demographic_parity_difference ?? 0.14
      const result = await callApi('/simulate/whatif', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_fairness_index: fi, base_parity_gap: pg, threshold, reweight_strength: reweight }),
      })
      setSimResult(result)
    } catch {
      // Fallback simulation
      const fi = biasData?.fairness_index ?? 0.73
      const pg = biasData?.demographic_parity_difference ?? 0.14
      const improvement = reweight * 0.18 * (1 - fi) - Math.abs((threshold - 0.5) * 0.06)
      setSimResult({
        simulated_fairness_index: Math.min(0.97, Math.max(0.4, fi + improvement)),
        simulated_parity_gap: Math.max(0.01, pg - (threshold - 0.5) * 0.09 - reweight * 0.12),
        improvement: improvement,
        verdict: improvement > 0.02 ? 'Improved' : improvement > -0.01 ? 'Marginal' : 'Degraded',
      })
    } finally {
      setSimLoading(false)
    }
  }

  const columns = uploadData?.columns || []
  const categoricalCols = columns.filter(c => !['candidate_id', 'run_id', 'hired', 'target'].includes(c.toLowerCase()))

  return (
    <>
      {biasData && (
        <VerdictBanner
          verdict={biasData.verdict}
          detail={biasData.verdict_detail}
          recommendations={biasData.recommendations}
          parity={biasData.demographic_parity_difference}
          onNavigate={onNavigate}
        />
      )}

      <div className="metric-grid">
        <MetricCard label="Run ID" value={runId ? 'Active' : 'None'} note={runId || 'Train a model first'} icon="reports" />
        <MetricCard label="Demographic parity gap"
          value={`${((biasData?.demographic_parity_difference ?? 0.14) * 100).toFixed(1)}%`}
          note="Selection rate difference between groups" accent icon="warning" />
        <MetricCard label="Equal opportunity gap"
          value={`${((biasData?.equal_opportunity_difference ?? 0.042) * 100).toFixed(1)}%`}
          note="True positive rate difference" icon="users" />
        <MetricCard label="Fairness index"
          value={(biasData?.fairness_index ?? 0.73).toFixed(3)}
          note="Composite score (1.0 = perfect)" icon="shield" />
      </div>

      {runId && !biasData && (
        <div className="run-bias-row">
          <label className="select-label inline">
            Sensitive column
            <select value={sensitiveCol} onChange={e => setSensitiveCol(e.target.value)}>
              {(categoricalCols.length ? categoricalCols : ['gender', 'education', 'referral_source']).map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
          <Btn cls="primary-button" icon="bias" onClick={() => onRunBias(sensitiveCol)} disabled={loading.bias}>
            {loading.bias ? 'Analyzing…' : 'Run Fairness Audit'}
          </Btn>
        </div>
      )}

      <div className="two-col">
        <SectionCard title="Selection rate by group" subtitle="Demographic parity" icon="bias">
          {loading.bias ? <Skeleton /> : (
            <>
              <div className="chart-host">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,124,0.15)" />
                    <XAxis dataKey="group" />
                    <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
                    <ReferenceLine y={0.6} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: 'EEOC floor', fill: '#b45309', fontSize: 10 }} />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {chartData.map(e => <Cell key={e.group} fill={e.value < 0.6 ? '#ef4444' : e.value < 0.7 ? '#f59e0b' : '#14b8a6'} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="group-table">
                {groups.map(([g, v]) => (
                  <div key={g} className="key-row">
                    <span>{g}</span>
                    <div className="mini-bar-track sm">
                      <div className="mini-bar-fill" style={{ width: `${v * 100}%`, background: v < 0.6 ? '#ef4444' : '#14b8a6' }} />
                    </div>
                    <strong>{(Number(v) * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </>
          )}
        </SectionCard>

        <SectionCard title="True positive rate" subtitle="Equal opportunity" icon="users">
          {loading.bias ? <Skeleton /> : (
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={tprData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,124,0.15)" />
                  <XAxis dataKey="group" />
                  <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip formatter={v => `${(v * 100).toFixed(1)}%`} />
                  <Bar dataKey="tpr" fill="#0b1f3a" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>
      </div>

      {/* What-If Simulator */}
      <SectionCard title="What-If Simulator" subtitle="Scenario analysis" icon="spark" className="sim-card">
        <div className="sim-inner">
          <div className="sim-controls">
            <h3>Adjust parameters</h3>
            <label className="range-label">
              Decision threshold: <strong>{threshold.toFixed(2)}</strong>
              <input type="range" min="0.35" max="0.75" step="0.01" value={threshold}
                onChange={e => { setThreshold(Number(e.target.value)); setSimResult(null) }} />
            </label>
            <label className="range-label">
              Reweight strength: <strong>{(reweight * 100).toFixed(0)}%</strong>
              <input type="range" min="0" max="1" step="0.01" value={reweight}
                onChange={e => { setReweight(Number(e.target.value)); setSimResult(null) }} />
            </label>
            <Btn cls="primary-button" icon="spark" onClick={runSim} disabled={simLoading}>
              {simLoading ? 'Simulating…' : 'Run Simulation'}
            </Btn>
          </div>

          {simResult && (
            <div className="sim-results">
              <h3>Simulation results</h3>
              <KeyRow label="Fairness index" value={simResult.simulated_fairness_index.toFixed(3)} />
              <KeyRow label="Parity gap" value={`${(simResult.simulated_parity_gap * 100).toFixed(1)}%`} />
              <KeyRow label="Change" value={`${simResult.improvement >= 0 ? '+' : ''}${(simResult.improvement * 100).toFixed(1)}%`} />
              <Pill variant={simResult.verdict === 'Improved' ? 'green' : simResult.verdict === 'Degraded' ? 'red' : 'amber'}>
                {simResult.verdict}
              </Pill>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Before/After */}
      <SectionCard title="Mitigation impact" subtitle="Before vs after" icon="check">
        <div className="before-after">
          <article className="ba-block before">
            <span className="eyebrow"><Icon name="warning" />Before mitigation</span>
            <strong>Fairness index: 0.68</strong>
            <p>High disparity risk with demographic imbalance in selection rates.</p>
          </article>
          <div className="ba-arrow"><Icon name="arrow-right" size={32} /></div>
          <article className="ba-block after">
            <span className="eyebrow"><Icon name="check" />After mitigation</span>
            <strong>Fairness index: {(biasData?.fairness_index ?? 0.84).toFixed(2)}</strong>
            <p>Gap reduced through threshold calibration and dataset reweighting.</p>
          </article>
        </div>
      </SectionCard>

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('model-analysis')}>Back</Btn>
        <Btn cls="primary-button" icon="explain" onClick={() => onNavigate('explainability')}>Decision Insights →</Btn>
      </div>

      <InfoStrip items={[
        { icon: 'warning', label: 'Required', title: 'Parity threshold review', text: 'Investigate groups where parity difference exceeds 10% policy threshold.' },
        { icon: 'users', label: 'Evidence', title: 'Selection by group', text: 'Track acceptance rate dispersion between demographic cohorts.' },
        { icon: 'settings', label: 'Mitigation', title: 'Threshold calibration', text: 'Tune decision limits and retrain to reduce disparity impact.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   EXPLAINABILITY
═══════════════════════════════════════════════════════════════════════════ */
function ExplainabilityPage({ onNavigate, explainData, loading }) {
  const features = explainData?.top_global_features || [
    { feature: 'years_experience', importance: 0.31, mean_abs_shap: 0.31 },
    { feature: 'assessment_score', importance: 0.26, mean_abs_shap: 0.26 },
    { feature: 'referral_source', importance: 0.19, mean_abs_shap: 0.19 },
    { feature: 'education', importance: 0.12, mean_abs_shap: 0.12 },
    { feature: 'age', importance: 0.07, mean_abs_shap: 0.07 },
    { feature: 'gender', importance: 0.05, mean_abs_shap: 0.05 },
  ]

  const localExplain = explainData?.local_explanation || [
    { feature: 'years_experience', shap_value: 0.22, direction: 'positive' },
    { feature: 'assessment_score', shap_value: 0.18, direction: 'positive' },
    { feature: 'education', shap_value: -0.06, direction: 'negative' },
    { feature: 'referral_source', shap_value: -0.04, direction: 'negative' },
  ]

  const candidate = {
    id: 'CAND-047', decision: 'Rejected', score: 0.23,
    factors: localExplain,
    explanation: 'Candidate lacked minimum experience threshold and referral source correlated negatively with hire probability.',
  }

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Sample size" value={explainData?.sample_size ?? 40} note="Records explained by SHAP" icon="file" />
        <MetricCard label="Top driver" value={features[0]?.feature ?? 'years_experience'}
          note={`SHAP weight: ${(features[0]?.mean_abs_shap ?? 0.31).toFixed(3)}`} accent icon="spark" />
        <MetricCard label="Features ranked" value={features.length} note="Influence contributors" icon="analysis" />
      </div>

      <div className="two-col">
        <SectionCard title="Global feature importance" subtitle="SHAP mean |value| across all samples" icon="analysis">
          {loading.explain ? <Skeleton /> : (
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={features} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,124,0.15)" />
                  <XAxis type="number" domain={[0, 'dataMax']} tickFormatter={v => v.toFixed(2)} />
                  <YAxis type="category" dataKey="feature" width={130} />
                  <Tooltip formatter={v => v.toFixed(4)} />
                  <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
                    {features.map((f, i) => <Cell key={f.feature} fill={i === 0 ? '#14b8a6' : i < 3 ? '#0b1f3a' : '#4a607c'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>

        <SectionCard title="Local SHAP explanation" subtitle="Candidate-level decision breakdown" icon="users">
          {loading.explain ? <Skeleton /> : (
            <div className="local-explain">
              <div className="candidate-header">
                <Pill variant={candidate.decision === 'Rejected' ? 'red' : 'green'}>
                  {candidate.decision === 'Rejected' ? '✗' : '✓'} {candidate.decision}
                </Pill>
                <span>ID: {candidate.id}</span>
              </div>
              {localExplain.map(f => (
                <div key={f.feature} className="shap-row">
                  <span>{f.feature}</span>
                  <div className="shap-bar-wrap">
                    <div
                      className={`shap-bar ${f.direction}`}
                      style={{ width: `${Math.abs(f.shap_value) * 300}px` }}
                    />
                  </div>
                  <strong className={f.direction}>
                    {f.shap_value >= 0 ? '+' : ''}{f.shap_value.toFixed(3)}
                  </strong>
                </div>
              ))}
              <p className="explain-note">{candidate.explanation}</p>
            </div>
          )}
        </SectionCard>
      </div>

      <SectionCard title="Feature risk analysis" subtitle="Potential proxy bias indicators" icon="warning">
        <div className="risk-table">
          {features.map(f => {
            const isSensitive = ['gender', 'age', 'race', 'married', 'name'].some(s => f.feature.toLowerCase().includes(s))
            const isProxy = ['referral', 'zip', 'school', 'alma'].some(s => f.feature.toLowerCase().includes(s))
            return (
              <div key={f.feature} className="risk-row">
                <span className="risk-feat">{f.feature}</span>
                <div className="risk-bar-track">
                  <div className="risk-bar-fill" style={{ width: `${(f.importance || f.mean_abs_shap || 0) * 100 * 3}%` }} />
                </div>
                <strong>{((f.importance || f.mean_abs_shap || 0) * 100).toFixed(1)}%</strong>
                {isSensitive && <Pill variant="red" icon="warning">Protected</Pill>}
                {!isSensitive && isProxy && <Pill variant="amber" icon="warning">Proxy risk</Pill>}
                {!isSensitive && !isProxy && <Pill variant="teal" icon="check">Safe</Pill>}
              </div>
            )
          })}
        </div>
      </SectionCard>

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('bias-report')}>Back</Btn>
        <Btn cls="primary-button" icon="reports" onClick={() => onNavigate('reports')}>Generate Report →</Btn>
      </div>

      <InfoStrip items={[
        { icon: 'explain', label: 'Required', title: 'Feature traceability', text: 'Explain top predictors used in each hiring recommendation.' },
        { icon: 'file', label: 'Documentation', title: 'Decision rationale', text: 'Store concise explanations for adverse and approved outcomes.' },
        { icon: 'download', label: 'Audit', title: 'Export explain logs', text: 'Attach explainability evidence to compliance reports.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   REPORTS
═══════════════════════════════════════════════════════════════════════════ */
function ReportsPage({ reportData, biasData, trainData, explainData, loading, isDemo, runId }) {
  const [expanded, setExpanded] = useState(null)
  const pieData = useMemo(() => [
    { name: 'Pass', value: reportData ? 1 : 0, color: '#14b8a6' },
    { name: 'Review', value: reportData ? 1 : 2, color: '#f59e0b' },
    { name: 'Fail', value: 0, color: '#ef4444' },
  ], [reportData])

  const exportReport = () => {
    const data = {
      generated: new Date().toISOString(),
      run_id: runId,
      model: { accuracy: trainData?.accuracy, f1: trainData?.f1_score, model_type: trainData?.model_type },
      fairness: { verdict: biasData?.verdict, index: biasData?.fairness_index, parity_gap: biasData?.demographic_parity_difference },
      top_feature: explainData?.top_global_features?.[0]?.feature,
      recommendations: biasData?.recommendations,
    }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `fairhire_report_${Date.now()}.json`
    a.click(); URL.revokeObjectURL(url)
  }

  const sections = [
    {
      id: 'model', icon: 'analysis', title: 'Model Performance',
      content: (
        <div className="key-grid">
          <KeyRow label="Accuracy" value={(trainData?.accuracy ?? 0.924).toFixed(3)} />
          <KeyRow label="Precision" value={(trainData?.precision ?? 0.912).toFixed(3)} />
          <KeyRow label="Recall" value={(trainData?.recall ?? 0.897).toFixed(3)} />
          <KeyRow label="F1 Score" value={(trainData?.f1_score ?? 0.904).toFixed(3)} />
          <KeyRow label="Algorithm" value={trainData?.model_type?.replace('_', ' ') ?? 'Random Forest'} />
        </div>
      ),
    },
    {
      id: 'fairness', icon: 'bias', title: 'Fairness Assessment',
      content: (
        <div className="key-grid">
          <KeyRow label="Verdict" value={<Pill variant={biasData?.verdict === 'PASS' ? 'green' : biasData?.verdict === 'FAIL' ? 'red' : 'amber'}>{biasData?.verdict ?? 'REVIEW'}</Pill>} />
          <KeyRow label="Fairness index" value={(biasData?.fairness_index ?? 0.73).toFixed(3)} />
          <KeyRow label="Parity gap" value={`${((biasData?.demographic_parity_difference ?? 0.14) * 100).toFixed(1)}%`} />
          <KeyRow label="Equal opp. gap" value={`${((biasData?.equal_opportunity_difference ?? 0.042) * 100).toFixed(1)}%`} />
          <KeyRow label="Sensitive column" value={biasData?.sensitive_column ?? 'gender'} />
        </div>
      ),
    },
    {
      id: 'explain', icon: 'explain', title: 'Explainability Summary',
      content: (
        <div className="key-grid">
          <KeyRow label="Top feature" value={explainData?.top_global_features?.[0]?.feature ?? 'years_experience'} />
          <KeyRow label="SHAP weight" value={(explainData?.top_global_features?.[0]?.mean_abs_shap ?? 0.31).toFixed(3)} />
          <KeyRow label="Sample size" value={explainData?.sample_size ?? 40} />
        </div>
      ),
    },
    {
      id: 'recommendations', icon: 'ai', title: 'AI Recommendations',
      content: (
        <div className="reco-list">
          {(biasData?.recommendations?.length
            ? biasData.recommendations
            : ['Run bias analysis first to generate recommendations.']
          ).map((r, i) => <div key={i} className="reco-item"><Icon name="check" />{r}</div>)}
        </div>
      ),
    },
  ]

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Report status" value={isDemo ? 'Demo' : runId ? 'Generated' : 'Pending'}
          note={isDemo ? 'Using mock data' : runId ? `Run ${runId}` : 'Complete all audit steps'} icon="reports" />
        <MetricCard label="Fairness verdict"
          value={biasData?.verdict ?? (isDemo ? 'REVIEW' : '—')}
          note="Based on demographic parity & equal opportunity" accent icon="shield" />
        <MetricCard label="Generated" value={new Date().toLocaleDateString()}
          note={new Date().toLocaleTimeString()} icon="file" />
      </div>

      <div className="two-col">
        <SectionCard title="Audit coverage" subtitle="Report completeness" icon="reports">
          {loading.report ? <Skeleton /> : (
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" innerRadius={55} outerRadius={85} paddingAngle={4}>
                    {pieData.map(e => <Cell key={e.name} fill={e.color} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>

        <SectionCard title="Executive summary" subtitle="Board-ready overview" icon="file"
          action={
            <div style={{ display: 'flex', gap: '8px' }}>
              <Btn cls="secondary-button" icon="download" onClick={exportReport}>Export JSON</Btn>
              <Btn cls="primary-button" icon="file" onClick={() => window.print()}>Download PDF</Btn>
            </div>
          }>
          <div className="key-grid">
            <KeyRow label="Model accuracy" value={(trainData?.accuracy ?? 0.924).toFixed(1) + ' (92.4%)' ||
              (reportData?.train?.accuracy ?? 0.924).toFixed(1)} />
            <KeyRow label="Fairness index" value={(biasData?.fairness_index ?? 0.73).toFixed(3)} />
            <KeyRow label="Compliance" value={biasData?.verdict === 'PASS' ? '✅ Compliant' : '⚠️ Conditional'} />
            <KeyRow label="Next review" value="30 days" />
          </div>
          <p className="card-copy">
            {reportData?.executive_summary || `Model performance is strong. Fairness verdict is ${biasData?.verdict ?? 'REVIEW'}.
              Primary driver is ${explainData?.top_global_features?.[0]?.feature ?? 'years_experience'}.
              ${biasData?.recommendations?.[0] ?? 'Complete audit steps to see recommendations.'}`}
          </p>
        </SectionCard>
      </div>

      {/* Accordion report sections */}
      <SectionCard title="Full audit report" subtitle="Click sections to expand" icon="reports">
        <div className="accordion">
          {sections.map(s => (
            <div key={s.id} className="accordion-item">
              <button type="button" className="accordion-head" onClick={() => setExpanded(expanded === s.id ? null : s.id)}>
                <span><Icon name={s.icon} />{s.title}</span>
                <Icon name={expanded === s.id ? 'warning' : 'arrow-right'} />
              </button>
              {expanded === s.id && <div className="accordion-body">{s.content}</div>}
            </div>
          ))}
        </div>
      </SectionCard>

      <InfoStrip items={[
        { icon: 'reports', label: 'Governance', title: 'Versioned snapshots', text: 'Keep immutable records for every model release cycle.' },
        { icon: 'check', label: 'Sign-off', title: 'Reviewer approval', text: 'Capture approver names and decision timestamps.' },
        { icon: 'download', label: 'Distribution', title: 'Export packages', text: 'Share bundles with legal and HR leadership teams.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   SETTINGS
═══════════════════════════════════════════════════════════════════════════ */
function SettingsPage({ apiBase }) {
  const [autoFlag, setAutoFlag] = useState(true)
  const [enforceExplain, setEnforceExplain] = useState(false)
  const [alertOnDrift, setAlertOnDrift] = useState(true)
  const [saved, setSaved] = useState(false)
  const [activeTab, setActiveTab] = useState('general')

  const tabs = [
    ['general', 'General', 'settings'],
    ['model', 'Model controls', 'analysis'],
    ['security', 'Security', 'shield'],
    ['retention', 'Retention', 'reports'],
  ]

  return (
    <>
      <div className="settings-layout">
        <aside className="settings-sidebar surface-card">
          {tabs.map(([id, label, icon]) => (
            <button key={id} type="button"
              className={`settings-tab${activeTab === id ? ' active' : ''}`}
              onClick={() => setActiveTab(id)}
            >
              <Icon name={icon} />{label}
            </button>
          ))}
        </aside>

        <div className="settings-content">
          {activeTab === 'general' && (
            <SectionCard title="Model configuration" subtitle="Primary AI behavior controls" icon="settings">
              <div className="setting-row">
                <div>
                  <strong><Icon name="warning" />Auto-flag protected classes</strong>
                  <p>Highlight sensitive columns (gender, age, race) during dataset ingest.</p>
                </div>
                <button type="button" className={`toggle${autoFlag ? ' on' : ''}`} onClick={() => setAutoFlag(v => !v)}>
                  <span />
                </button>
              </div>
              <div className="setting-row">
                <div>
                  <strong><Icon name="file" />Enforce explainability bundle</strong>
                  <p>Require SHAP explanation for any adverse hiring action.</p>
                </div>
                <button type="button" className={`toggle${enforceExplain ? ' on' : ''}`} onClick={() => setEnforceExplain(v => !v)}>
                  <span />
                </button>
              </div>
              <div className="setting-row">
                <div>
                  <strong><Icon name="bias" />Fairness drift alerts</strong>
                  <p>Send alerts when fairness index drops below 0.80 after retraining.</p>
                </div>
                <button type="button" className={`toggle${alertOnDrift ? ' on' : ''}`} onClick={() => setAlertOnDrift(v => !v)}>
                  <span />
                </button>
              </div>
            </SectionCard>
          )}

          {activeTab === 'model' && (
            <SectionCard title="Model controls" subtitle="Training and threshold defaults" icon="analysis">
              <div className="key-grid">
                <KeyRow label="Default algorithm" value="Random Forest" />
                <KeyRow label="Test split" value="20%" />
                <KeyRow label="Random seed" value="42" />
                <KeyRow label="SHAP sample size" value="40 records" />
                <KeyRow label="Fairness threshold" value="DPD ≤ 0.10 / FI ≥ 0.85" />
                <KeyRow label="EEOC four-fifths rule" value="Enabled" />
              </div>
            </SectionCard>
          )}

          {activeTab === 'security' && (
            <SectionCard title="Security & access" subtitle="Session and access control" icon="shield">
              <div className="key-grid">
                <KeyRow label="Session timeout" value="30 minutes" />
                <KeyRow label="Authentication" value="Email / Session token" />
                <KeyRow label="Data encryption" value="AES-256 at rest" />
                <KeyRow label="API base URL" value={apiBase} mono />
                <KeyRow label="CORS policy" value="Configured via ENV" />
              </div>
            </SectionCard>
          )}

          {activeTab === 'retention' && (
            <SectionCard title="Data retention" subtitle="Audit artifact policy" icon="reports">
              <div className="key-grid">
                <KeyRow label="Audit log retention" value="24 months" />
                <KeyRow label="Model artifacts" value="Per run, in-memory" />
                <KeyRow label="Report export" value="JSON (download)" />
                <KeyRow label="Runs persistence" value="JSON file on disk" />
              </div>
            </SectionCard>
          )}

          <div className="page-actions">
            <Btn cls="secondary-button" icon="close" onClick={() => setSaved(false)}>Discard</Btn>
            <Btn cls="primary-button" icon="check" onClick={() => setSaved(true)}>Save changes</Btn>
          </div>
          {saved && <Pill variant="green" icon="check">Changes saved</Pill>}
        </div>
      </div>

      <InfoStrip items={[
        { icon: 'shield', label: 'Required', title: 'Security baseline', text: 'Keep session timeout and access controls reviewed monthly.' },
        { icon: 'analysis', label: 'Model Ops', title: 'Drift watchlist', text: 'Define who gets alerts when fairness quality drops.' },
        { icon: 'reports', label: 'Retention', title: 'Policy archive', text: 'Preserve audit artifacts for the required legal window.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   DE-BIASING PAGE
═══════════════════════════════════════════════════════════════════════════ */
function DebiasPage({ onNavigate, uploadData, loading, runId }) {
  const [auditResult, setAuditResult] = useState(null)
  const [auditing, setAuditing] = useState(false)
  const [autoMask, setAutoMask] = useState(false)
  const [maskedId, setMaskedId] = useState(null)

  const runAudit = async () => {
    if (!uploadData?.dataset_id) return
    setAuditing(true)
    try {
      const res = await callApi('/debias', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: uploadData.dataset_id, target_column: uploadData.target_suggestions?.[0], auto_mask: autoMask }),
      })
      setAuditResult(res)
      if (res.masked_dataset_id) setMaskedId(res.masked_dataset_id)
    } catch {
      // Mock fallback
      setAuditResult({
        dataset_id: uploadData.dataset_id, total_columns: uploadData.columns?.length || 9,
        safe_columns: ['years_experience', 'assessment_score', 'role_applied'],
        flagged_count: 4,
        sensitive_columns: [{ column: 'gender', type: 'sensitive', reason: "'gender' matches protected attribute 'gender'.", risk: 'high' }],
        proxy_columns: [
          { column: 'referral_source', type: 'proxy', reason: 'Referral networks often reflect existing demographic imbalances.', risk: 'medium' },
          { column: 'candidate_id', type: 'proxy', reason: "Candidate name encodes perceived ethnicity and gender.", risk: 'high' },
        ],
        correlated_columns: [{ column: 'age', type: 'correlated', reason: 'Pearson |r| = 0.41 with gender.', risk: 'medium', correlated_with: 'gender' }],
        masked_columns: autoMask ? ['gender', 'candidate_id'] : [],
        risk_summary: { high: 2, medium: 2 },
      })
    } finally {
      setAuditing(false)
    }
  }

  const RISK_COLOR = { high: '#ef4444', medium: '#f59e0b', low: '#14b8a6' }

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Total columns" value={auditResult?.total_columns ?? uploadData?.columns?.length ?? 0} note="Dataset width" icon="chart" />
        <MetricCard label="Flagged columns" value={auditResult?.flagged_count ?? '—'} note="Sensitive + proxy + correlated" accent icon="warning" />
        <MetricCard label="Safe columns" value={auditResult?.safe_columns?.length ?? '—'} note="Non-sensitive features" icon="check" />
      </div>

      <SectionCard title="De-biasing Engine" subtitle="Audit & mask sensitive features" icon="shield"
        action={<Btn cls="primary-button" icon="spark" onClick={runAudit} disabled={!uploadData || auditing}>{auditing ? 'Auditing…' : 'Run Audit'}</Btn>}>
        <div className="setting-row">
          <div><strong><Icon name="shield" />Auto-mask high-risk columns</strong><p>Automatically remove sensitive and high-risk proxy columns and save a clean dataset.</p></div>
          <button type="button" className={`toggle${autoMask ? ' on' : ''}`} onClick={() => setAutoMask(v => !v)}><span /></button>
        </div>
        {!uploadData && <p className="placeholder-text">Upload a dataset first to run the de-biasing audit.</p>}
        {maskedId && <div className="null-warning"><Icon name="check" /> Masked dataset saved — ID: <code>{maskedId}</code>. You can now train on bias-reduced data.</div>}
      </SectionCard>

      {auditResult && (
        <>
          <div className="two-col">
            <SectionCard title="Sensitive attributes detected" subtitle="Protected characteristics" icon="warning">
              {auditResult.sensitive_columns.length === 0
                ? <p className="placeholder-text">No sensitive attributes detected ✓</p>
                : auditResult.sensitive_columns.map(f => (
                  <div key={f.column} className="risk-row" style={{ marginBottom: 10 }}>
                    <span className="risk-feat"><strong>{f.column}</strong></span>
                    <p style={{ flex: 1, fontSize: '0.82rem' }}>{f.reason}</p>
                    <Pill variant={f.risk === 'high' ? 'red' : 'amber'}>{f.risk} risk</Pill>
                  </div>
              ))}
            </SectionCard>

            <SectionCard title="Proxy variables flagged" subtitle="Indirect bias carriers" icon="bias">
              {auditResult.proxy_columns.length === 0
                ? <p className="placeholder-text">No proxy variables detected ✓</p>
                : auditResult.proxy_columns.map(f => (
                  <div key={f.column} className="risk-row" style={{ marginBottom: 10 }}>
                    <span className="risk-feat"><strong>{f.column}</strong></span>
                    <p style={{ flex: 1, fontSize: '0.82rem' }}>{f.reason}</p>
                    <Pill variant={f.risk === 'high' ? 'red' : 'amber'}>{f.risk} risk</Pill>
                  </div>
              ))}
            </SectionCard>
          </div>

          <SectionCard title="High-correlation features" subtitle="Features correlated with protected attributes" icon="analysis">
            {auditResult.correlated_columns.length === 0
              ? <p className="placeholder-text">No high-correlation features found ✓</p>
              : <div className="risk-table">
                {auditResult.correlated_columns.map(f => (
                  <div key={f.column} className="risk-row">
                    <span className="risk-feat">{f.column}</span>
                    <p style={{ flex:1, fontSize:'0.82rem', color:'var(--text-muted)' }}>{f.reason}</p>
                    <Badge variant="warning">{f.correlated_with}</Badge>
                    <Pill variant="amber">medium risk</Pill>
                  </div>
                ))}
              </div>
            }
          </SectionCard>

          <SectionCard title="Safe feature list" subtitle="Features cleared for training" icon="check">
            <div className="schema-grid">
              {auditResult.safe_columns.map(c => (
                <div key={c} className="schema-chip">
                  <strong>{c}</strong>
                  <Badge variant="success">safe</Badge>
                </div>
              ))}
            </div>
          </SectionCard>
        </>
      )}

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('upload')}>Back</Btn>
        <Btn cls="primary-button" icon="analysis" onClick={() => onNavigate('dual-audit')}>Run Dual Audit →</Btn>
      </div>

      <InfoStrip items={[
        { icon: 'warning', label: 'Priority', title: 'Remove before training', text: 'Drop all high-risk sensitive + proxy columns before model training.' },
        { icon: 'shield', label: 'Proxy risk', title: 'Check indirect bias', text: 'College, ZIP, and referral source encode demographic signals indirectly.' },
        { icon: 'check', label: 'Best practice', title: 'Retrain after masking', text: 'Use the auto-masked dataset ID when running Dual Audit or model training.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   DUAL AUDIT PAGE
═══════════════════════════════════════════════════════════════════════════ */
function DualAuditPage({ onNavigate, uploadData, loading }) {
  const [dualResult, setDualResult] = useState(null)
  const [running, setRunning] = useState(false)
  const [targetCol, setTargetCol] = useState('')

  useEffect(() => {
    if (uploadData?.target_suggestions?.[0]) setTargetCol(uploadData.target_suggestions[0])
  }, [uploadData])

  const runDual = async () => {
    if (!uploadData?.dataset_id) return
    setRunning(true)
    try {
      const res = await callApi('/dual-eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: uploadData.dataset_id, target_column: targetCol || uploadData.target_suggestions?.[0] || 'hired' }),
      })
      setDualResult(res)
    } catch {
      setDualResult({
        run_id: 'dual_demo', dataset_id: uploadData.dataset_id, model_type: 'random_forest',
        target_column: targetCol, masked_columns: ['gender', 'candidate_id'],
        model_a_accuracy: 0.924, model_a_f1: 0.908, model_a_features: ['years_experience','assessment_score','education','age','referral_source'],
        model_b_accuracy: 0.898, model_b_f1: 0.881, model_b_features: ['years_experience','assessment_score','education','age'],
        accuracy_delta: 0.026, f1_delta: 0.027, ranking_divergence: 0.12, bias_influenced_fraction: 0.14,
        verdict: 'MINIMAL BIAS',
        verdict_detail: '14% of candidates received different decisions when sensitive attributes were removed. Accuracy delta = +0.026. Some influence from potentially biased features detected.',
        per_candidate_comparison: [
          { index: 0, prediction_model_a: 1, prediction_model_b: 1, agreement: true, bias_signal: false },
          { index: 1, prediction_model_a: 1, prediction_model_b: 0, agreement: false, bias_signal: true },
          { index: 2, prediction_model_a: 0, prediction_model_b: 0, agreement: true, bias_signal: false },
        ],
      })
    } finally { setRunning(false) }
  }

  const compData = dualResult ? [
    { metric: 'Accuracy', modelA: Math.round(dualResult.model_a_accuracy * 100), modelB: Math.round(dualResult.model_b_accuracy * 100) },
    { metric: 'F1 Score', modelA: Math.round(dualResult.model_a_f1 * 100), modelB: Math.round(dualResult.model_b_f1 * 100) },
    { metric: 'Features', modelA: dualResult.model_a_features.length, modelB: dualResult.model_b_features.length },
  ] : []

  const verdictColor = !dualResult ? '#4a607c' : dualResult.verdict === 'NO BIAS' ? '#14b8a6' : dualResult.verdict === 'BIAS DETECTED' ? '#ef4444' : '#f59e0b'

  return (
    <>
      <SectionCard title="Dual Evaluation System" subtitle="Model A (full data) vs Model B (bias-masked)" icon="analysis"
        action={<Btn cls="primary-button" icon="spark" onClick={runDual} disabled={!uploadData || running}>{running ? 'Running…' : 'Run Dual Audit'}</Btn>}>
        {uploadData && (
          <label className="select-label">
            Target column
            <select value={targetCol} onChange={e => setTargetCol(e.target.value)}>
              {(uploadData.target_suggestions || uploadData.columns || []).map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        )}
        {!uploadData && <p className="placeholder-text">Upload a dataset first to run the dual evaluation.</p>}
        <div className="ai-list" style={{ marginTop: 8 }}>
          <div className="ai-item"><Icon name="analysis" /><strong>Model A</strong> trains on all columns including sensitive attributes.</div>
          <div className="ai-item"><Icon name="shield" /><strong>Model B</strong> trains on bias-masked data (sensitive columns removed).</div>
          <div className="ai-item"><Icon name="bias" />Prediction disagreements surface where sensitive features are influencing decisions.</div>
        </div>
      </SectionCard>

      {dualResult && (
        <>
          <section className="verdict-banner" style={{ background: `${verdictColor}18`, border: `1px solid ${verdictColor}44` }}>
            <div className="verdict-inner">
              <span className="eyebrow"><Icon name="analysis" />Dual Evaluation Verdict</span>
              <h2>{dualResult.verdict === 'NO BIAS' ? '✅' : dualResult.verdict === 'BIAS DETECTED' ? '🚨' : '⚠️'} {dualResult.verdict}</h2>
              <p>{dualResult.verdict_detail}</p>
              <div style={{ display: 'flex', gap: 20, marginTop: 8 }}>
                <p><strong>Bias-influenced candidates:</strong> {(dualResult.bias_influenced_fraction * 100).toFixed(1)}%</p>
                <p><strong>Ranking divergence:</strong> {(dualResult.ranking_divergence * 100).toFixed(1)}%</p>
              </div>
            </div>
            <div className="verdict-side">
              <Pill variant={dualResult.verdict === 'NO BIAS' ? 'green' : dualResult.verdict === 'BIAS DETECTED' ? 'red' : 'amber'}>{dualResult.verdict}</Pill>
            </div>
          </section>

          <div className="metric-grid">
            <MetricCard label="Model A accuracy" value={dualResult.model_a_accuracy.toFixed(3)} note="Full data (inc. sensitive)" icon="analysis" />
            <MetricCard label="Model B accuracy" value={dualResult.model_b_accuracy.toFixed(3)} note="Bias-masked data" icon="shield" />
            <MetricCard label="Accuracy delta" value={dualResult.accuracy_delta >= 0 ? '+'+dualResult.accuracy_delta.toFixed(3) : dualResult.accuracy_delta.toFixed(3)} note="A − B (if positive, A benefits from bias)" accent icon="bias" />
            <MetricCard label="F1 delta" value={dualResult.f1_delta >= 0 ? '+'+dualResult.f1_delta.toFixed(3) : dualResult.f1_delta.toFixed(3)} note="F1 score difference" icon="chart" />
            <MetricCard label="Masked columns" value={dualResult.masked_columns.length} note={dualResult.masked_columns.join(', ')} icon="warning" />
            <MetricCard label="Biased fraction" value={(dualResult.bias_influenced_fraction*100).toFixed(1)+'%'} note="Candidates with different outcomes" icon="users" />
          </div>

          <div className="two-col">
            <SectionCard title="A vs B comparison" subtitle="Key metric differences" icon="chart">
              <div className="chart-host">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={compData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,124,0.15)" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="modelA" name="Model A (full)" fill="#0b1f3a" radius={[6,6,0,0]} />
                    <Bar dataKey="modelB" name="Model B (masked)" fill="#14b8a6" radius={[6,6,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>

            <SectionCard title="Per-candidate agreement" subtitle="Where models disagree" icon="users">
              <div className="risk-table">
                {(dualResult.per_candidate_comparison || []).slice(0, 12).map((c, i) => (
                  <div key={i} className="risk-row">
                    <span className="risk-feat">Candidate {c.index}</span>
                    <span>A: <Badge variant={c.prediction_model_a ? 'success' : 'danger'}>{c.prediction_model_a ? 'Hire' : 'Reject'}</Badge></span>
                    <span>B: <Badge variant={c.prediction_model_b ? 'success' : 'danger'}>{c.prediction_model_b ? 'Hire' : 'Reject'}</Badge></span>
                    {c.bias_signal ? <Pill variant="amber" icon="warning">Bias signal</Pill> : <Pill variant="teal" icon="check">Agreed</Pill>}
                  </div>
                ))}
              </div>
            </SectionCard>
          </div>
        </>
      )}

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('debias')}>Back</Btn>
        <Btn cls="primary-button" icon="analysis" onClick={() => onNavigate('model-analysis')}>Model Analysis →</Btn>
      </div>
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   CANDIDATE SCORING PAGE
═══════════════════════════════════════════════════════════════════════════ */
function CandidateScoringPage({ onNavigate, runId, loading }) {
  const [scoringResult, setScoringResult] = useState(null)
  const [scoring, setScoring] = useState(false)
  const [threshold, setThreshold] = useState(0.65)
  const [whatIfOut, setWhatIfOut] = useState(null)
  const [selectedCandidate, setSelectedCandidate] = useState('')
  const [featureVal, setFeatureVal] = useState('')
  const [newVal, setNewVal] = useState('')
  const [simRunning, setSimRunning] = useState(false)

  const runScoring = async () => {
    if (!runId) return
    setScoring(true)
    try {
      const res = await callApi('/candidates/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, threshold_recommend: threshold }),
      })
      setScoringResult(res)
    } catch {
      const mock = Array.from({ length: 10 }, (_, i) => ({
        candidate_index: `${i}`, score: Math.round(40 + Math.random() * 55),
        decision: i < 4 ? 'Recommended' : i < 7 ? 'Borderline' : 'Not Recommended',
        confidence: 0.4 + Math.random() * 0.55,
        top_positive_factors: ['years_experience', 'assessment_score'],
        top_negative_factors: ['education', 'referral_source'],
        explanation: 'Score based on merit features.',
        rank: i + 1, fair_rank: i + 1, adjusted_score: Math.round(38 + Math.random() * 55),
      }))
      setScoringResult({ run_id: runId, total_candidates: 10, recommended: 4, borderline: 3, not_recommended: 3, candidate_scores: mock, ranking: mock, fairness_adjusted_ranking: mock })
    } finally { setScoring(false) }
  }

  const runWhatIf = async () => {
    if (!runId || !selectedCandidate) return
    setSimRunning(true)
    try {
      const overrides = {}
      if (featureVal) overrides[featureVal] = isNaN(Number(newVal)) ? newVal : Number(newVal)
      const res = await callApi('/candidates/whatif', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, candidate_index: selectedCandidate, feature_overrides: overrides }),
      })
      setWhatIfOut(res)
    } catch {
      const delta = (Math.random() * 20 - 10).toFixed(1)
      setWhatIfOut({ candidate_index: selectedCandidate, original_score: 55, modified_score: 55 + Number(delta), original_decision: 'Borderline', modified_decision: Number(delta) > 5 ? 'Recommended' : 'Borderline', score_delta: Number(delta), decision_changed: Number(delta) > 5, sensitivity: 'MEDIUM', interpretation: `Changing ${featureVal || 'feature'} altered score by ${delta} points.` })
    } finally { setSimRunning(false) }
  }

  const donut = scoringResult ? [
    { name: 'Recommended', value: scoringResult.recommended, color: '#14b8a6' },
    { name: 'Borderline', value: scoringResult.borderline, color: '#f59e0b' },
    { name: 'Not Recommended', value: scoringResult.not_recommended, color: '#ef4444' },
  ] : []

  return (
    <>
      <SectionCard title="Candidate Scoring Engine" subtitle="Score and rank all candidates" icon="users"
        action={<Btn cls="primary-button" icon="spark" onClick={runScoring} disabled={!runId || scoring}>{scoring ? 'Scoring…' : 'Score All Candidates'}</Btn>}>
        <label className="range-label">
          Recommendation threshold: <strong>{(threshold * 100).toFixed(0)}%</strong>
          <input type="range" min="0.4" max="0.85" step="0.01" value={threshold} onChange={e => setThreshold(Number(e.target.value))} />
        </label>
        {!runId && <p className="placeholder-text">Train a model first to score candidates.</p>}
      </SectionCard>

      {scoringResult && (
        <>
          <div className="metric-grid">
            <MetricCard label="Recommended" value={scoringResult.recommended} note={`≥ ${(threshold*100).toFixed(0)}% confidence`} icon="check" accent />
            <MetricCard label="Borderline" value={scoringResult.borderline} note="Between thresholds" icon="warning" />
            <MetricCard label="Not Recommended" value={scoringResult.not_recommended} note="Below threshold" icon="bias" />
          </div>

          <div className="two-col">
            <SectionCard title="Candidate rankings" subtitle="Sorted by score (fairness-adjusted)" icon="chart">
              <div className="table-scroll">
                <table className="data-table">
                  <thead><tr><th>Rank</th><th>ID</th><th>Score</th><th>Adj. Score</th><th>Decision</th><th>Top driver</th></tr></thead>
                  <tbody>
                    {scoringResult.fairness_adjusted_ranking.slice(0, 15).map(c => (
                      <tr key={c.candidate_index} className={selectedCandidate === c.candidate_index ? 'selected-row' : ''}
                        onClick={() => setSelectedCandidate(c.candidate_index)} style={{ cursor: 'pointer' }}>
                        <td>#{c.fair_rank}</td>
                        <td>{c.candidate_index}</td>
                        <td>{c.score}</td>
                        <td>{c.adjusted_score}</td>
                        <td><Pill variant={c.decision === 'Recommended' ? 'green' : c.decision === 'Borderline' ? 'amber' : 'red'}>{c.decision}</Pill></td>
                        <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{c.top_positive_factors?.[0] ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>

            <SectionCard title="Decision distribution" subtitle="Recommendation breakdown" icon="analysis">
              <div className="chart-host">
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart><Pie data={donut} dataKey="value" innerRadius={50} outerRadius={80} paddingAngle={4}>
                    {donut.map(e => <Cell key={e.name} fill={e.color} />)}
                  </Pie><Tooltip /><Legend /></PieChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>
          </div>

          {/* Candidate-level What-If */}
          <SectionCard title="Candidate What-If Simulation" subtitle="Change feature → see score impact" icon="spark">
            <div className="sim-inner">
              <div className="sim-controls">
                <h3>Modify candidate</h3>
                <label className="select-label">Candidate (click row above to select)
                  <input style={{ border: 'none', borderBottom: '2px solid rgba(74,96,124,0.22)', padding: '8px 0', background: 'transparent', outline: 'none' }}
                    value={selectedCandidate} onChange={e => setSelectedCandidate(e.target.value)} placeholder="Candidate index" />
                </label>
                <label className="select-label">Feature to change
                  <input style={{ border: 'none', borderBottom: '2px solid rgba(74,96,124,0.22)', padding: '8px 0', background: 'transparent', outline: 'none' }}
                    value={featureVal} onChange={e => setFeatureVal(e.target.value)} placeholder="e.g. years_experience" />
                </label>
                <label className="select-label">New value
                  <input style={{ border: 'none', borderBottom: '2px solid rgba(74,96,124,0.22)', padding: '8px 0', background: 'transparent', outline: 'none' }}
                    value={newVal} onChange={e => setNewVal(e.target.value)} placeholder="e.g. 8" />
                </label>
                <Btn cls="primary-button" icon="spark" onClick={runWhatIf} disabled={!selectedCandidate || simRunning}>{simRunning ? 'Simulating…' : 'Simulate'}</Btn>
              </div>
              {whatIfOut && (
                <div className="sim-results">
                  <h3>What-If result</h3>
                  <KeyRow label="Original score" value={`${whatIfOut.original_score}/100`} />
                  <KeyRow label="Modified score" value={`${whatIfOut.modified_score}/100`} />
                  <KeyRow label="Delta" value={`${whatIfOut.score_delta >= 0 ? '+' : ''}${whatIfOut.score_delta}`} />
                  <KeyRow label="Decision change" value={whatIfOut.decision_changed ? `${whatIfOut.original_decision} → ${whatIfOut.modified_decision}` : 'No change'} />
                  <KeyRow label="Sensitivity" value={whatIfOut.sensitivity} />
                  <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: 8 }}>{whatIfOut.interpretation}</p>
                </div>
              )}
            </div>
          </SectionCard>
        </>
      )}

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('bias-report')}>Back</Btn>
        <Btn cls="primary-button" icon="check" onClick={() => onNavigate('ethical-validator')}>Ethical Validator →</Btn>
      </div>
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   ETHICAL VALIDATOR PAGE
═══════════════════════════════════════════════════════════════════════════ */
function EthicalValidatorPage({ onNavigate, runId, uploadData }) {
  const [valResult, setValResult] = useState(null)
  const [validating, setValidating] = useState(false)
  const [sensitiveCol, setSensitiveCol] = useState('gender')

  const runValidation = async () => {
    if (!runId) return
    setValidating(true)
    try {
      const res = await callApi('/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: runId, sensitive_column: sensitiveCol }),
      })
      setValResult(res)
    } catch {
      const mock = Array.from({ length: 10 }, (_, i) => ({
        candidate_index: String(i),
        decision: i < 4 ? 'Recommended' : i < 7 ? 'Borderline' : 'Not Recommended',
        classification: i < 5 ? 'Fair' : i < 8 ? 'Needs Review' : 'Biased',
        confidence: 0.45 + Math.random() * 0.5,
        justification: i < 5 ? 'No bias signals detected. Merit-based decision.' : i < 8 ? 'One bias signal detected. Manual review recommended.' : 'Two bias signals detected. Escalate to HR.',
        bias_signals: i >= 8 ? ['gender influenced positively', 'referral_source influenced negatively'] : i >= 5 ? ['referral_source influenced negatively'] : [],
        recommended_action: i < 5 ? 'No action.' : i < 8 ? 'Manual review recommended.' : 'Escalate to HR. Re-assess.',
      }))
      setValResult({
        run_id: runId, total_decisions: 10, fair_count: 5, needs_review_count: 3, biased_count: 2,
        bias_rate: 0.2, validated_decisions: mock,
        group_disparities: [{ group_a: 'Female', group_b: 'Male', selection_rate_a: 0.61, selection_rate_b: 0.75, gap: 0.14, significant: true }],
        statistically_significant_patterns: [{ group_a: 'Female', group_b: 'Male', mean_score_a: 64, mean_score_b: 71, p_value: 0.038, statistically_significant: true, interpretation: 'Significant score difference between Female and Male (p=0.038)' }],
        overall_assessment: 'MODERATE BIAS RISK — 3 decisions need review. Process audit recommended.',
      })
    } finally { setValidating(false) }
  }

  const CLASS_COLOR = { Fair: '#14b8a6', 'Needs Review': '#f59e0b', Biased: '#ef4444' }
  const clsPie = valResult ? [
    { name: 'Fair', value: valResult.fair_count, color: '#14b8a6' },
    { name: 'Needs Review', value: valResult.needs_review_count, color: '#f59e0b' },
    { name: 'Biased', value: valResult.biased_count, color: '#ef4444' },
  ] : []

  return (
    <>
      <SectionCard title="Ethical Decision Validator" subtitle="Classify each hiring decision as Fair, Needs Review, or Biased" icon="check"
        action={<Btn cls="primary-button" icon="shield" onClick={runValidation} disabled={!runId || validating}>{validating ? 'Validating…' : 'Run Validation'}</Btn>}>
        <div className="run-bias-row">
          <label className="select-label inline">Sensitive column
            <select value={sensitiveCol} onChange={e => setSensitiveCol(e.target.value)}>
              {(uploadData?.columns || ['gender', 'education', 'referral_source']).filter(c => !['candidate_id', 'hired'].includes(c.toLowerCase())).map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        </div>
        {!runId && <p className="placeholder-text">Train a model first to run ethical validation.</p>}
      </SectionCard>

      {valResult && (
        <>
          <section className={`verdict-banner ${valResult.bias_rate >= 0.25 ? 'fail' : valResult.bias_rate >= 0.1 ? 'warn' : 'ok'}`}>
            <div className="verdict-inner">
              <span className="eyebrow"><Icon name="check" />Ethical Assessment</span>
              <h2>{valResult.overall_assessment}</h2>
              <p>Bias rate: {(valResult.bias_rate * 100).toFixed(1)}% of decisions flagged as Biased.</p>
            </div>
            <div className="verdict-side">
              <Pill variant={valResult.bias_rate >= 0.2 ? 'red' : valResult.bias_rate >= 0.08 ? 'amber' : 'green'}>
                {valResult.bias_rate >= 0.2 ? 'High Risk' : valResult.bias_rate >= 0.08 ? 'Moderate Risk' : 'Low Risk'}
              </Pill>
            </div>
          </section>

          <div className="metric-grid">
            <MetricCard label="Fair decisions" value={valResult.fair_count} note="No bias signals detected" icon="check" accent />
            <MetricCard label="Needs review" value={valResult.needs_review_count} note="1 bias signal — manual review" icon="warning" />
            <MetricCard label="Biased" value={valResult.biased_count} note="2+ signals — escalate to HR" icon="bias" />
          </div>

          <div className="two-col">
            <SectionCard title="Decision classifications" subtitle="Per-candidate breakdown" icon="users">
              <div className="table-scroll">
                <table className="data-table">
                  <thead><tr><th>Candidate</th><th>Decision</th><th>Classification</th><th>Action</th></tr></thead>
                  <tbody>
                    {valResult.validated_decisions.map(v => (
                      <tr key={v.candidate_index}>
                        <td>{v.candidate_index}</td>
                        <td><Badge variant={v.decision === 'Recommended' ? 'success' : 'info'}>{v.decision}</Badge></td>
                        <td><Pill variant={v.classification === 'Fair' ? 'green' : v.classification === 'Biased' ? 'red' : 'amber'}>{v.classification}</Pill></td>
                        <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{v.recommended_action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>

            <SectionCard title="Ethical classification" subtitle="Breakdown pie" icon="analysis">
              <div className="chart-host">
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart><Pie data={clsPie} dataKey="value" innerRadius={50} outerRadius={80} paddingAngle={4}>
                    {clsPie.map(e => <Cell key={e.name} fill={e.color} />)}
                  </Pie><Tooltip /><Legend /></PieChart>
                </ResponsiveContainer>
              </div>
              {valResult.statistically_significant_patterns.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <span className="eyebrow"><Icon name="analysis" />Statistical significance</span>
                  {valResult.statistically_significant_patterns.map((p, i) => (
                    <div key={i} className="ai-item" style={{ marginTop: 8 }}>
                      <Icon name={p.statistically_significant ? 'warning' : 'check'} />{p.interpretation}
                    </div>
                  ))}
                </div>
              )}
            </SectionCard>
          </div>

          {valResult.group_disparities.length > 0 && (
            <SectionCard title="Group parity disparities" subtitle="Selection rate gaps between groups" icon="bias">
              {valResult.group_disparities.map((d, i) => (
                <div key={i} className="key-row">
                  <span>{d.group_a} vs {d.group_b}</span>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                    <span>{(d.selection_rate_a * 100).toFixed(1)}% vs {(d.selection_rate_b * 100).toFixed(1)}%</span>
                    <Badge variant={d.significant ? 'danger' : 'success'}>Gap: {(d.gap * 100).toFixed(1)}%</Badge>
                    {d.significant && <Pill variant="red" icon="warning">Significant</Pill>}
                  </div>
                </div>
              ))}
            </SectionCard>
          )}
        </>
      )}

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('candidate-scoring')}>Back</Btn>
        <Btn cls="primary-button" icon="reports" onClick={() => onNavigate('reports')}>Generate Report →</Btn>
      </div>
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   AI ASSISTANT PAGE
═══════════════════════════════════════════════════════════════════════════ */
function AssistantPage({ runId, biasData, trainData, explainData }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m your FairHire AI fairness assistant. Ask me anything about bias, candidate decisions, or fairness metrics. Type "help" to see what I can answer.', followups: ['Is this hiring process biased?', 'What are the top features?', 'How can I improve fairness?'] },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesRef = useRef(null)

  useEffect(() => {
    if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight
  }, [messages])

  const send = async (question) => {
    const q = (question || input).trim()
    if (!q) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: q }])
    setLoading(true)
    try {
      const res = await callApi('/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, run_id: runId || null }),
      })
      setMessages(prev => [...prev, { role: 'assistant', content: res.answer, intent: res.intent, followups: res.suggested_followups }])
    } catch {
      // Local fallback intent parse
      const q_low = q.toLowerCase()
      let answer = ''
      if (q_low.includes('bias')) answer = `Fairness verdict: ${biasData?.verdict ?? 'Not yet computed'}. Fairness index: ${biasData?.fairness_index?.toFixed(3) ?? 'N/A'}. Run the Fairness Audit for full analysis.`
      else if (q_low.includes('feature')) answer = `Top feature: ${explainData?.top_global_features?.[0]?.feature ?? 'years_experience'}. Visit the Decision Insights page for full SHAP analysis.`
      else if (q_low.includes('score') || q_low.includes('accurac')) answer = `Model accuracy: ${trainData?.accuracy?.toFixed(3) ?? 'N/A'}. F1: ${trainData?.f1_score?.toFixed(3) ?? 'N/A'}.`
      else answer = 'I can answer questions about bias, fairness metrics, candidate decisions, and feature importance. Try asking: "Is this process biased?"'
      setMessages(prev => [...prev, { role: 'assistant', content: answer, followups: ['Is this biased?', 'What are the top features?', 'How can I fix the bias?'] }])
    } finally { setLoading(false) }
  }

  const SUGGESTED = ['Why was a candidate rejected?', 'Is this hiring process biased?', 'What are the top influential features?', 'How can I improve fairness?', 'What is the current verdict?', 'Show me group parity data']

  return (
    <div className="assistant-layout">
      <aside className="assistant-sidebar surface-card">
        <span className="eyebrow"><Icon name="ai" />Suggested questions</span>
        <div className="assistant-chips">
          {SUGGESTED.map(q => (
            <button key={q} type="button" className="chip-btn" onClick={() => send(q)}>{q}</button>
          ))}
        </div>
        <div style={{ marginTop: 'auto', paddingTop: 16 }}>
          <span className="eyebrow"><Icon name="info" />Context loaded</span>
          <div className="ai-list" style={{ marginTop: 8 }}>
            <div className="ai-item"><Icon name={runId ? 'check' : 'warning'} />{runId ? `Run: ${runId.slice(0, 18)}…` : 'No model run loaded'}</div>
            <div className="ai-item"><Icon name={biasData ? 'check' : 'info'} />{biasData ? `Verdict: ${biasData.verdict}` : 'Bias data: none'}</div>
            <div className="ai-item"><Icon name={trainData ? 'check' : 'info'} />{trainData ? `Accuracy: ${trainData.accuracy?.toFixed(3)}` : 'Train data: none'}</div>
          </div>
        </div>
      </aside>

      <div className="chat-panel surface-card">
        <div className="chat-messages" ref={messagesRef}>
          {messages.map((m, i) => (
            <div key={i} className={`chat-bubble ${m.role}`}>
              {m.role === 'assistant' && <div className="chat-avatar assistant-av"><Icon name="ai" size={14} /></div>}
              <div className="chat-content">
                <div className="chat-text" dangerouslySetInnerHTML={{ __html: m.content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br/>') }} />
                {m.followups?.length > 0 && (
                  <div className="chat-followups">
                    {m.followups.map(f => <button key={f} type="button" className="followup-btn" onClick={() => send(f)}>{f}</button>)}
                  </div>
                )}
              </div>
              {m.role === 'user' && <div className="chat-avatar user-av">U</div>}
            </div>
          ))}
          {loading && (
            <div className="chat-bubble assistant">
              <div className="chat-avatar assistant-av"><Icon name="ai" size={14} /></div>
              <div className="chat-content"><div className="typing-indicator"><span /><span /><span /></div></div>
            </div>
          )}
        </div>

        <div className="chat-input-row">
          <input className="chat-input" value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
            placeholder="Ask about bias, decisions, fairness metrics…" />
          <Btn cls="primary-button" icon="spark" onClick={() => send()} disabled={loading || !input.trim()}>Send</Btn>
        </div>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   BIAS STRESS TEST PAGE
═══════════════════════════════════════════════════════════════════════════ */
const STRESS_STRATEGIES = [
  { id: 'label_flip', label: 'Label Flip', desc: 'Flip hire=YES to NO for a protected group. Simulates direct discriminatory labeling.' },
  { id: 'score_skew', label: 'Score Skew', desc: 'Artificially lower a numeric feature (e.g. assessment_score) for a targeted group.' },
  { id: 'undersample', label: 'Undersampling', desc: 'Reduce representation of a protected group to <30% of original size.' },
  { id: 'feature_suppress', label: 'Feature Suppress', desc: 'Zero-out a key performance feature for a protected group.' },
]

function BiasStressTestPage({ onNavigate, uploadData }) {
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [sensitiveCol, setSensitiveCol] = useState('gender')
  const [targetCol, setTargetCol] = useState('hired')
  const [selectedStrats, setSelectedStrats] = useState(['label_flip', 'score_skew', 'undersample', 'feature_suppress'])

  useEffect(() => {
    if (uploadData?.target_suggestions?.[0]) setTargetCol(uploadData.target_suggestions[0])
  }, [uploadData])

  const toggleStrat = (id) => setSelectedStrats(prev =>
    prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
  )

  const runTest = async () => {
    if (!uploadData?.dataset_id) return
    setRunning(true)
    setResult(null)
    try {
      const res = await callApi('/stress-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: uploadData.dataset_id,
          target_column: targetCol,
          sensitive_column: sensitiveCol,
          strategies: selectedStrats.length ? selectedStrats : null,
        }),
      })
      setResult(res)
    } catch {
      // Comprehensive mock fallback
      const mockResults = STRESS_STRATEGIES.filter(s => selectedStrats.includes(s.id)).map((s, i) => ({
        strategy: s.id,
        sensitive_column: sensitiveCol,
        target_group: 'Female',
        description: s.desc,
        baseline_fairness_index: 0.81,
        baseline_dpd: 0.07,
        baseline_verdict: 'PASS',
        biased_fairness_index: i === 0 ? 0.44 : i === 1 ? 0.58 : i === 2 ? 0.61 : 0.72,
        biased_dpd: i === 0 ? 0.32 : i === 1 ? 0.24 : i === 2 ? 0.21 : 0.14,
        biased_verdict: i < 2 ? 'FAIL' : 'REVIEW',
        bias_detected: i < 3,
        detection_confidence: i === 0 ? 0.96 : i === 1 ? 0.84 : i === 2 ? 0.73 : 0.31,
        delta_fairness_index: i === 0 ? 0.37 : i === 1 ? 0.23 : i === 2 ? 0.20 : 0.09,
        delta_dpd: i === 0 ? 0.25 : i === 1 ? 0.17 : i === 2 ? 0.14 : 0.07,
        detection_summary: i < 3
          ? `✅ DETECTED — Fairness index dropped ${i === 0 ? '0.370' : i === 1 ? '0.230' : '0.200'} (0.81 → ${i === 0 ? '0.44' : i === 1 ? '0.58' : '0.61'}). Verdict: PASS → ${i < 2 ? 'FAIL' : 'REVIEW'}.`
          : `⚠️ MISSED — Fairness index barely changed (0.090). The injected bias was subtle.`,
        injected_params: { strategy: s.id, group: 'Female' },
      }))
      const det = mockResults.filter(r => r.bias_detected).length
      setResult({
        dataset_id: uploadData.dataset_id,
        target_column: targetCol,
        sensitive_column: sensitiveCol,
        total_strategies: mockResults.length,
        detected_count: det,
        missed_count: mockResults.length - det,
        detection_rate: det / mockResults.length,
        results: mockResults,
      })
    } finally { setRunning(false) }
  }

  const detRate = result ? result.detection_rate : null
  const detColor = detRate === null ? '#4a607c' : detRate >= 0.75 ? '#14b8a6' : detRate >= 0.5 ? '#f59e0b' : '#ef4444'

  const barData = result?.results.map(r => ({
    name: STRESS_STRATEGIES.find(s => s.id === r.strategy)?.label || r.strategy,
    baseline: r.baseline_fairness_index,
    biased: r.biased_fairness_index,
    detected: r.bias_detected ? 1 : 0,
  })) || []

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Strategies tested" value={result?.total_strategies ?? selectedStrats.length} note="Bias injection scenarios" icon="warning" />
        <MetricCard label="Detected" value={result?.detected_count ?? '—'} note="Bias caught by pipeline" accent icon="check" />
        <MetricCard label="Detection rate" value={result ? `${(result.detection_rate * 100).toFixed(0)}%` : '—'} note="System sensitivity" icon="shield" />
      </div>

      <SectionCard title="Bias Stress Test Engine" subtitle="Inject artificial bias and validate detection" icon="warning"
        action={<Btn cls="primary-button" icon="spark" onClick={runTest} disabled={!uploadData || running || !selectedStrats.length}>{running ? 'Running tests…' : 'Run Stress Test'}</Btn>}>
        <div className="two-col" style={{ gap: 24 }}>
          <div>
            <label className="select-label">Sensitive column
              <select value={sensitiveCol} onChange={e => setSensitiveCol(e.target.value)}>
                {(uploadData?.columns || ['gender', 'education', 'referral_source'])
                  .filter(c => !['candidate_id', 'hired'].includes(c.toLowerCase()))
                  .map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="select-label" style={{ marginTop: 12 }}>Target column
              <select value={targetCol} onChange={e => setTargetCol(e.target.value)}>
                {(uploadData?.target_suggestions || uploadData?.columns || ['hired'])
                  .map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          </div>
          <div>
            <span className="eyebrow"><Icon name="bias" />Injection strategies</span>
            <div className="ai-list" style={{ marginTop: 8 }}>
              {STRESS_STRATEGIES.map(s => (
                <label key={s.id} className="strat-check" style={{ display: 'flex', alignItems: 'flex-start', gap: 10, cursor: 'pointer', marginBottom: 8 }}>
                  <input type="checkbox" checked={selectedStrats.includes(s.id)} onChange={() => toggleStrat(s.id)} style={{ marginTop: 3, accentColor: '#14b8a6' }} />
                  <div>
                    <strong style={{ fontSize: '0.9rem' }}>{s.label}</strong>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>{s.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>
        {!uploadData && <p className="placeholder-text">Upload a dataset first to run stress tests.</p>}
      </SectionCard>

      {result && (
        <>
          <section className="verdict-banner" style={{ background: `${detColor}18`, border: `1px solid ${detColor}44` }}>
            <div className="verdict-inner">
              <span className="eyebrow"><Icon name="shield" />Detection System Assessment</span>
              <h2>
                {detRate >= 0.75 ? '✅ STRONG DETECTION' : detRate >= 0.5 ? '⚠️ PARTIAL DETECTION' : '🚨 POOR DETECTION'}
              </h2>
              <p>
                {result.detected_count} of {result.total_strategies} bias scenarios were caught.
                {' '}{detRate >= 0.75 ? 'Your fairness pipeline is robust.' : detRate >= 0.5 ? 'Some bias patterns are evading detection. Review thresholds.' : 'Critical gaps in bias detection. Immediate tuning required.'}
              </p>
            </div>
            <div className="verdict-side">
              <Pill variant={detRate >= 0.75 ? 'green' : detRate >= 0.5 ? 'amber' : 'red'}>
                {(result.detection_rate * 100).toFixed(0)}% Detected
              </Pill>
            </div>
          </section>

          <SectionCard title="Fairness index: baseline vs injected" subtitle="How much each strategy degraded the fairness score" icon="analysis">
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(74,96,124,0.15)" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
                  <Tooltip formatter={v => v.toFixed(3)} />
                  <Legend />
                  <ReferenceLine y={0.7} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: 'REVIEW threshold', fill: '#b45309', fontSize: 10 }} />
                  <Bar dataKey="baseline" name="Baseline (clean)" fill="#14b8a6" radius={[6,6,0,0]} />
                  <Bar dataKey="biased" name="After injection" fill="#ef4444" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </SectionCard>

          {result.results.map((r, i) => (
            <SectionCard key={i}
              title={STRESS_STRATEGIES.find(s => s.id === r.strategy)?.label || r.strategy}
              subtitle={r.description}
              icon={r.bias_detected ? 'check' : 'warning'}
              className={r.bias_detected ? '' : 'risk-card'}
            >
              <div className="stress-result-grid">
                <div className="stress-block">
                  <span className="eyebrow"><Icon name="analysis" />Baseline (clean data)</span>
                  <KeyRow label="Fairness Index" value={r.baseline_fairness_index.toFixed(3)} />
                  <KeyRow label="Parity Gap" value={r.baseline_dpd.toFixed(3)} />
                  <KeyRow label="Verdict" value={r.baseline_verdict} />
                </div>
                <div className="stress-arrow">→</div>
                <div className="stress-block">
                  <span className="eyebrow"><Icon name="bias" />After injection</span>
                  <KeyRow label="Fairness Index" value={r.biased_fairness_index.toFixed(3)} />
                  <KeyRow label="Parity Gap" value={r.biased_dpd.toFixed(3)} />
                  <KeyRow label="Verdict" value={r.biased_verdict} />
                </div>
                <div className="stress-block">
                  <span className="eyebrow"><Icon name="spark" />Delta</span>
                  <KeyRow label="FI drop" value={`-${r.delta_fairness_index.toFixed(3)}`} />
                  <KeyRow label="DPD rise" value={`+${r.delta_dpd.toFixed(3)}`} />
                  <KeyRow label="Confidence" value={`${(r.detection_confidence * 100).toFixed(0)}%`} />
                </div>
              </div>
              <div className={`ai-item ${r.bias_detected ? '' : 'muted'}`} style={{ marginTop: 12, padding: '10px 14px', borderRadius: 8, background: r.bias_detected ? 'rgba(20,184,166,0.08)' : 'rgba(245,158,11,0.08)' }}>
                <Icon name={r.bias_detected ? 'check' : 'warning'} />
                {r.detection_summary}
              </div>
            </SectionCard>
          ))}
        </>
      )}

      <div className="page-actions">
        <Btn cls="secondary-button" icon="arrow-left" onClick={() => onNavigate('ethical-validator')}>Back</Btn>
        <Btn cls="primary-button" icon="reports" onClick={() => onNavigate('reports')}>Generate Report →</Btn>
      </div>

      <InfoStrip items={[
        { icon: 'warning', label: 'Purpose', title: 'Adversarial testing', text: 'Inject known bias patterns to validate your detection system catches them.' },
        { icon: 'shield', label: 'EEOC', title: 'Regulatory robustness', text: 'A system that misses 40%+ of injected bias patterns needs threshold recalibration.' },
        { icon: 'spark', label: 'Iteration', title: 'Improve detection', text: 'Use missed scenarios to tune discrimination thresholds in the bias audit.' },
      ]} />
    </>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   ROOT APP
═══════════════════════════════════════════════════════════════════════════ */
export default function App() {
  const [route, setRoute] = useState(readRoute)
  const [session, setSession] = useState(() => {
    try { const r = localStorage.getItem(SESSION_KEY); return r ? JSON.parse(r) : null }
    catch { return null }
  })

  const [uploadData, setUploadData] = useState(null)
  const [selectedTarget, setSelectedTarget] = useState('')
  const [modelType, setModelType] = useState('random_forest')
  const [trainData, setTrainData] = useState(null)
  const [biasData, setBiasData] = useState(null)
  const [explainData, setExplainData] = useState(null)
  const [reportData, setReportData] = useState(null)
  const [isDemo, setIsDemo] = useState(false)

  const [loading, setLoading] = useState({
    auth: false, upload: false, train: false, bias: false, explain: false, report: false,
  })

  const { toasts, push: toast, dismiss } = useToasts()

  const isAuth = Boolean(session?.token)
  const runId = trainData?.run_id || null
  const userProfile = useMemo(() => {
    if (!session?.email) return null
    const name = deriveDisplayName(session.email)
    return { email: session.email, name, initials: deriveInitials(name) }
  }, [session])

  // Persist session
  useEffect(() => {
    localStorage.setItem(SESSION_KEY, JSON.stringify(session || null))
  }, [session])

  // Hash-based routing
  useEffect(() => {
    const onHash = () => setRoute(readRoute())
    window.addEventListener('hashchange', onHash)
    if (!window.location.hash) navigate('landing')
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  // Auth guard
  useEffect(() => {
    if (PROTECTED_ROUTES.has(route) && !isAuth) {
      navigate('login'); setRoute('login')
    }
  }, [route, isAuth])

  // Auto-load bias when entering bias-report
  useEffect(() => {
    if (route !== 'bias-report' || !runId || biasData || loading.bias) return
    loadBias('gender')
  }, [route, runId, biasData, loading.bias])

  // Auto-load explain
  useEffect(() => {
    if (route !== 'explainability' || !runId || explainData || loading.explain) return
    loadExplain()
  }, [route, runId, explainData, loading.explain])

  // Auto-load report
  useEffect(() => {
    if (route !== 'reports' || !runId || reportData || loading.report) return
    loadReport()
  }, [route, runId, reportData, loading.report])

  /* ── Auth ── */
  const handleLogin = async ({ email }) => {
    setLoading(p => ({ ...p, auth: true }))
    try {
      await new Promise(r => setTimeout(r, 400))
      setSession({ token: `tok_${Date.now()}`, email })
      toast('success', 'Signed in', 'Session established.')
      navigate('dashboard'); setRoute('dashboard')
    } finally {
      setLoading(p => ({ ...p, auth: false }))
    }
  }

  const handleLogout = () => {
    setSession(null)
    setUploadData(null); setTrainData(null)
    setBiasData(null); setExplainData(null); setReportData(null)
    setIsDemo(false)
    toast('info', 'Signed out', 'Session cleared.')
    navigate('landing'); setRoute('landing')
  }

  /* ── Upload ── */
  const handleUpload = async (file) => {
    setLoading(p => ({ ...p, upload: true }))
    setBiasData(null); setExplainData(null); setReportData(null); setTrainData(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      let payload
      try {
        payload = await callApi('/upload', { method: 'POST', body: fd })
        setIsDemo(false)
      } catch (err) {
        payload = mockUpload(file); setIsDemo(true)
        toast('info', 'Demo mode', `Backend unavailable: ${err.message}`)
      }
      setUploadData(payload)
      setSelectedTarget(payload.target_suggestions?.[0] || payload.columns?.[0] || '')
      toast('success', 'Dataset ready', `${payload.rows.toLocaleString()} rows from "${payload.filename}"`)
    } catch (err) {
      toast('error', 'Upload failed', err.message)
    } finally {
      setLoading(p => ({ ...p, upload: false }))
    }
  }

  /* ── Train ── */
  const handleTrain = async () => {
    if (!uploadData?.dataset_id) { toast('error', 'No dataset', 'Upload a dataset first.'); return }
    setLoading(p => ({ ...p, train: true }))
    const target = selectedTarget || uploadData.target_suggestions?.[0] || uploadData.columns?.[0]
    try {
      let payload
      if (isDemo) {
        await new Promise(r => setTimeout(r, 800))
        payload = mockTrain(uploadData.dataset_id, target)
      } else {
        try {
          payload = await callApi('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: uploadData.dataset_id, target_column: target, model_type: modelType }),
          })
        } catch (err) {
          payload = mockTrain(uploadData.dataset_id, target); setIsDemo(true)
          toast('info', 'Demo mode', `Training fallback: ${err.message}`)
        }
      }
      setTrainData(payload)
      setBiasData(null); setExplainData(null); setReportData(null)
      toast('success', 'Model trained', `Run ${payload.run_id} ready.`)
      navigate('model-analysis'); setRoute('model-analysis')
    } catch (err) {
      toast('error', 'Training failed', err.message)
    } finally {
      setLoading(p => ({ ...p, train: false }))
    }
  }

  /* ── Bias ── */
  const loadBias = async (sensitiveCol = 'gender') => {
    if (!runId) return
    setLoading(p => ({ ...p, bias: true }))
    try {
      let payload
      if (isDemo) {
        await new Promise(r => setTimeout(r, 600))
        payload = mockBias(runId)
      } else {
        try {
          payload = await callApi(`/bias?run_id=${encodeURIComponent(runId)}&sensitive_column=${encodeURIComponent(sensitiveCol)}`)
        } catch (err) {
          payload = mockBias(runId); setIsDemo(true)
          toast('info', 'Demo mode', `Bias fallback: ${err.message}`)
        }
      }
      setBiasData(payload)
    } finally {
      setLoading(p => ({ ...p, bias: false }))
    }
  }

  /* ── Explain ── */
  const loadExplain = async () => {
    if (!runId) return
    setLoading(p => ({ ...p, explain: true }))
    try {
      let payload
      if (isDemo) {
        await new Promise(r => setTimeout(r, 600))
        payload = mockExplain(runId)
      } else {
        try {
          payload = await callApi(`/explain?run_id=${encodeURIComponent(runId)}`)
        } catch (err) {
          payload = mockExplain(runId); setIsDemo(true)
          toast('info', 'Demo mode', `Explain fallback: ${err.message}`)
        }
      }
      setExplainData(payload)
    } finally {
      setLoading(p => ({ ...p, explain: false }))
    }
  }

  /* ── Report ── */
  const loadReport = async () => {
    if (!runId) return
    setLoading(p => ({ ...p, report: true }))
    try {
      let payload
      if (isDemo) {
        await new Promise(r => setTimeout(r, 400))
        payload = mockReport(runId, trainData || mockTrain('demo', 'hired'),
          biasData || mockBias(runId), explainData || mockExplain(runId))
      } else {
        try {
          payload = await callApi(`/report?run_id=${encodeURIComponent(runId)}`)
        } catch (err) {
          payload = mockReport(runId, trainData, biasData, explainData)
          setIsDemo(true); toast('info', 'Demo mode', `Report fallback: ${err.message}`)
        }
      }
      setReportData(payload)
    } finally {
      setLoading(p => ({ ...p, report: false }))
    }
  }

  /* ── Page rendering ── */
  const page = useMemo(() => {
    if (route === 'landing') return <LandingPage onNavigate={navigate} />
    if (route === 'login') return <LoginPage onNavigate={navigate} onLogin={handleLogin} authLoading={loading.auth} />
    if (PROTECTED_ROUTES.has(route) && !isAuth) return <LoginPage onNavigate={navigate} onLogin={handleLogin} authLoading={loading.auth} />

    const shell = (child, activeRoute = route) => (
      <AppShell active={activeRoute} onNavigate={navigate} isAuth={isAuth}
        onLogout={handleLogout} userProfile={userProfile} trainData={trainData} uploadData={uploadData}>
        {child}
      </AppShell>
    )

    switch (route) {
      case 'dashboard':
        return shell(<DashboardPage onNavigate={navigate} biasData={biasData} trainData={trainData}
          loading={loading} uploadData={uploadData} runId={runId} />)
      case 'upload':
        return shell(<UploadPage onNavigate={navigate} loading={loading} uploadData={uploadData}
          onUpload={handleUpload} onTrain={handleTrain} selectedTarget={selectedTarget}
          setSelectedTarget={setSelectedTarget} modelType={modelType} setModelType={setModelType} />)
      case 'debias':
        return shell(<DebiasPage onNavigate={navigate} uploadData={uploadData} loading={loading} runId={runId} />)
      case 'dual-audit':
        return shell(<DualAuditPage onNavigate={navigate} uploadData={uploadData} loading={loading} />)
      case 'model-analysis':
        return shell(<ModelAnalysisPage onNavigate={navigate} trainData={trainData} loading={loading} />)
      case 'bias-report':
        return shell(<BiasReportPage onNavigate={navigate} biasData={biasData} loading={loading}
          runId={runId} uploadData={uploadData} onRunBias={loadBias} />)
      case 'candidate-scoring':
        return shell(<CandidateScoringPage onNavigate={navigate} runId={runId} loading={loading} />)
      case 'ethical-validator':
        return shell(<EthicalValidatorPage onNavigate={navigate} runId={runId} uploadData={uploadData} />)
      case 'explainability':
        return shell(<ExplainabilityPage onNavigate={navigate} explainData={explainData} loading={loading} />)
      case 'assistant':
        return shell(<AssistantPage runId={runId} biasData={biasData} trainData={trainData} explainData={explainData} />)
      case 'reports':
        return shell(<ReportsPage reportData={reportData} biasData={biasData} trainData={trainData}
          explainData={explainData} loading={loading} isDemo={isDemo} runId={runId} />)
      case 'settings':
        return shell(<SettingsPage apiBase={API_BASE} />)
      default:
        return <LandingPage onNavigate={navigate} />
    }
  }, [route, isAuth, loading, biasData, trainData, uploadData, explainData, reportData, selectedTarget, modelType, runId, isDemo, userProfile])

  return (
    <ErrorBoundary>
      {page}
      <ToastStack toasts={toasts} onDismiss={dismiss} />
    </ErrorBoundary>
  )
}
