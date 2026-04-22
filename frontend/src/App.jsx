import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  saveDatasetUpload,
  saveGeneratedReport,
  saveTrainingRun,
  upsertUserProfile,
} from './firestoreService'

const ROUTES = new Set([
  'landing',
  'login',
  'dashboard',
  'upload',
  'model-analysis',
  'bias-report',
  'explainability',
  'reports',
  'settings',
])

const PROTECTED_ROUTES = new Set(['dashboard', 'upload', 'model-analysis', 'bias-report', 'explainability', 'reports', 'settings'])
const SESSION_KEY = 'fairhire_session'
const THEME_KEY = 'fairhire_theme_mode'
const IS_LOCAL_HOST = typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname)
const API_BASE = (import.meta.env.VITE_API_URL || (IS_LOCAL_HOST ? 'http://127.0.0.1:8000' : '')).replace(/\/$/, '')
const API_CONFIG_ERROR = 'Backend API is not configured for production. Set VITE_API_URL to your deployed backend URL and redeploy the frontend.'
const ROUTE_META = {
  dashboard: ['Workspace', 'Dashboard'],
  upload: ['Workspace', 'Upload Dataset'],
  'model-analysis': ['Workspace', 'Model Evaluation'],
  'bias-report': ['Workspace', 'Fairness Audit'],
  explainability: ['Workspace', 'Decision Rationale'],
  reports: ['Workspace', 'Reports'],
  settings: ['Workspace', 'Settings'],
  login: ['Access', 'Sign In'],
  landing: ['Welcome', 'Overview'],
}

function readRoute() {
  const value = window.location.hash.replace(/^#\/?/, '') || 'landing'
  return ROUTES.has(value) ? value : 'landing'
}

function navigate(route) {
  window.location.hash = `#/${route}`
}

function deriveDisplayName(email) {
  if (!email || !email.includes('@')) return 'FairHire User'
  const localPart = email.split('@')[0]
  return localPart
    .replace(/[._-]+/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function deriveInitials(name) {
  const parts = (name || 'FairHire User').split(' ').filter(Boolean)
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase()
  return `${parts[0][0] || ''}${parts[1][0] || ''}`.toUpperCase()
}

function Icon({ name }) {
  const className = name === 'shield' || name === 'check'
    ? 'icon-svg icon-animated'
    : 'icon-svg'

  switch (name) {
    case 'dashboard':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M4 13.5V4h7v9.5H4Zm9 6.5V11h7v9h-7ZM4 20v-4.5h7V20H4Zm9-12V4h7v4h-7Z" /></svg>
    case 'upload':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M12 3 6.5 8.5l1.4 1.4L11 6.8V16h2V6.8l3.1 3.1 1.4-1.4L12 3ZM5 18v2h14v-2H5Z" /></svg>
    case 'analysis':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M4 19h16v2H4v-2Zm2-3 3-5 3 2 4-7 2 1.2-5.4 9-3-2-2.1 3.5L6 16Z" /></svg>
    case 'bias':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M12 3 2.8 20h18.4L12 3Zm0 4.8 5.5 10.2H6.5L12 7.8Zm-1 3.2h2v4h-2v-4Zm0 5h2v2h-2v-2Z" /></svg>
    case 'explain':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M12 2 2 7v10l10 5 10-5V7L12 2Zm0 2.3 7.9 4L12 12.3 4.1 8.3 12 4.3ZM4 18V9.9l8 4v8.1l-8-4Zm16 0-8 4v-8.1l8-4V18Z" /></svg>
    case 'reports':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M6 3h9l5 5v13H6V3Zm8 1.5V9h4.5L14 4.5ZM8 12h8v2H8v-2Zm0 4h8v2H8v-2Zm0-8h3v2H8V8Z" /></svg>
    case 'settings':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="m19.14 12.94 1.2-1.2-1.9-3.3-1.62.56a6.7 6.7 0 0 0-1.16-.67L15.5 6h-3l-.16 1.33c-.4.17-.8.39-1.16.67l-1.62-.56-1.9 3.3 1.2 1.2c-.05.3-.08.62-.08.94s.03.64.08.94l-1.2 1.2 1.9 3.3 1.62-.56c.36.28.76.5 1.16.67L12.5 18h3l.16-1.33c.4-.17.8-.39 1.16-.67l1.62.56 1.9-3.3-1.2-1.2c.05-.3.08-.62.08-.94s-.03-.64-.08-.94ZM12 15.2a3.2 3.2 0 1 1 0-6.4 3.2 3.2 0 0 1 0 6.4Z" /></svg>
    case 'search':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="m21 20-4.3-4.3a7 7 0 1 0-1.4 1.4L20 21l1-1ZM5 11a6 6 0 1 1 12 0A6 6 0 0 1 5 11Z" /></svg>
    case 'spark':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="m12 2 1.8 5.1L19 9l-5.2 1.9L12 16l-1.8-5.1L5 9l5.2-1.9L12 2Zm7 9 1.2 3.4L24 16l-3.8 1.6L19 21l-1.2-3.4L14 16l3.8-1.6L19 11Z" /></svg>
    case 'download':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M12 3v10.2l3.6-3.6 1.4 1.4L12 17 6.9 11 8.3 9.6 12 13.2V3ZM5 19h14v2H5v-2Z" /></svg>
    case 'arrow-left':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="m12 5-1.4 1.4 3.6 3.6H5v2h9.2l-3.6 3.6L12 17l7-7-7-5Z" /></svg>
    case 'check':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="m9.2 16.2-4-4L3.8 13l5.4 5.4L20.2 7.4 18.8 6 9.2 16.2Z" /></svg>
    case 'warning':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M12 3 2.8 20h18.4L12 3Zm1 13h-2v-2h2v2Zm0-3h-2V8h2v5Z" /></svg>
    case 'users':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm9 1a3 3 0 1 0 0-6 3 3 0 0 0 0 6ZM2 21v-1a6 6 0 0 1 12 0v1H2Zm14 0v-1.2a5.5 5.5 0 0 0-1.2-3.4A7 7 0 0 1 22 21v1h-6Z" /></svg>
    case 'shield':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="m12 2 8 3v6c0 5.2-3.1 8.9-8 11-4.9-2.1-8-5.8-8-11V5l8-3Zm0 2.1L6 6.4V11c0 4.1 2.4 7.1 6 8.8 3.6-1.7 6-4.7 6-8.8V6.4l-6-2.3Z" /></svg>
    case 'login':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M10 17v-2h7V9h-7V7h9v10h-9ZM6 19V5h2v14H6Zm6.3-4.3L11 13.4 12.6 12H4v-2h8.6L11 8.6 12.3 7.3 16.9 12l-4.6 4.7Z" /></svg>
    case 'logout':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M14 17v-2h3V9h-3V7h5v10h-5ZM6 19V5h7v2H8v10h5v2H6Zm7.4-4.2-1.4-1.4 2.4-2.4H3v-2h11.4L12 6.6l1.4-1.4 4.8 4.8-4.8 4.8Z" /></svg>
    case 'file':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M6 2h9l5 5v15H6V2Zm8 1.5V8h4.5L14 3.5ZM8 12h8v2H8v-2Zm0 4h8v2H8v-2Zm0-8h3v2H8V8Z" /></svg>
    case 'menu':
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M4 7h16v2H4V7Zm0 5h16v2H4v-2Zm0 5h16v2H4v-2Z" /></svg>
    default:
      return <svg viewBox="0 0 24 24" aria-hidden="true" className={className}><path d="M4 12h16" /></svg>
  }
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unknown rendering error' }
  }

  componentDidCatch(error) {
    console.error('UI render error:', error)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="app-error-screen">
          <h1>Something went wrong</h1>
          <p>{this.state.message}</p>
          <button
            type="button"
            className="primary-button"
            onClick={() => {
              this.setState({ hasError: false, message: '' })
              window.location.hash = '#/landing'
            }}
          >
            Reload Interface
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

function LabelIcon({ icon, children }) {
  return (
    <span className="label-icon">
      <Icon name={icon} />
      <span>{children}</span>
    </span>
  )
}

function ButtonWithIcon({ type = 'button', className, icon, children, onClick, disabled = false }) {
  return (
    <button type={type} className={className} onClick={onClick} disabled={disabled}>
      <Icon name={icon} />
      {children}
    </button>
  )
}

function ToastStack({ toasts, onDismiss }) {
  return (
    <div className="toast-stack">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast ${toast.type}`}>
          <Icon name={toast.type === 'error' ? 'warning' : toast.type === 'success' ? 'check' : 'spark'} />
          <div>
            <strong>{toast.title}</strong>
            <p>{toast.message}</p>
            {toast.insight ? <span className="toast-insight">{toast.insight}</span> : null}
          </div>
          <button type="button" className="toast-close" onClick={() => onDismiss(toast.id)}>
            ×
          </button>
        </div>
      ))}
    </div>
  )
}

function Skeleton({ className = '' }) {
  return <div className={`skeleton ${className}`.trim()} />
}

function GlobalLoadingOverlay({ visible, label }) {
  if (!visible) return null

  return (
    <div className="global-loader-overlay" role="status" aria-live="polite" aria-label={label}>
      <div className="global-loader-card surface-glass">
        <div className="loader-orbit" aria-hidden="true">
          <span className="loader-ring outer" />
          <span className="loader-ring inner" />
        </div>
        <strong>{label}</strong>
        <p>Preparing a smooth, data-rich experience.</p>
      </div>
    </div>
  )
}

function AppShell({ active, onNavigate, actions, children, isAuthenticated, onLogout, userProfile, isSidebarCompact, onToggleSidebar, loading }) {
  const navItems = [
    ['dashboard', 'Dashboard', 'dashboard'],
    ['upload', 'Upload Dataset', 'upload'],
    ['model-analysis', 'Model Evaluation', 'analysis'],
    ['bias-report', 'Fairness Audit', 'bias'],
    ['explainability', 'Decision Rationale', 'explain'],
    ['reports', 'Reports', 'reports'],
    ['settings', 'Settings', 'settings'],
  ]

  const flowSteps = [
    ['Upload Data', active !== 'upload' && isAuthenticated],
    ['Train Model', ['model-analysis', 'bias-report', 'explainability', 'reports', 'settings'].includes(active)],
    ['Audit Bias', ['bias-report', 'explainability', 'reports', 'settings'].includes(active)],
    ['Review Rationale', ['explainability', 'reports', 'settings'].includes(active)],
    ['Export Report', ['reports', 'settings'].includes(active)],
  ]

  const crumbs = ROUTE_META[active] || ['Workspace', 'Overview']
  const hasBackgroundLoad = Boolean(loading?.upload || loading?.train || loading?.bias || loading?.explain || loading?.report)
  const loadLabel = loading?.upload
    ? 'Uploading dataset'
    : loading?.train
      ? 'Training model'
      : loading?.bias
        ? 'Computing fairness metrics'
        : loading?.explain
          ? 'Generating explainability'
          : loading?.report
            ? 'Building report'
            : ''

  return (
    <div className={isSidebarCompact ? 'app-shell compact' : 'app-shell'}>
      <div className="mesh-layer mesh-a" aria-hidden="true" />
      <div className="mesh-layer mesh-b" aria-hidden="true" />
      <aside className="sidebar surface-panel">
        <div className="brand-block">
          <div className="brand-mark" />
          <div className="brand-copy">
            <strong>FairHire AI</strong>
            <span>Architectural Curator</span>
          </div>
        </div>
        <nav>
          {navItems.map(([route, label, icon]) => (
            <button
              key={route}
              type="button"
              className={route === active ? 'nav-item active' : 'nav-item'}
              onClick={() => onNavigate(route)}
              title={label}
            >
              <Icon name={icon} />
              <span className="nav-label">{label}</span>
            </button>
          ))}
        </nav>
        {isAuthenticated && userProfile ? (
          <section className="profile-panel">
            <div className="profile-avatar" aria-hidden="true">{userProfile.initials}</div>
            <div className="profile-meta">
              <strong>{userProfile.name}</strong>
              <small>{userProfile.email}</small>
              <span className="profile-role"><Icon name="shield" />Audit Manager</span>
            </div>
          </section>
        ) : null}
        {!isAuthenticated ? (
          <ButtonWithIcon type="button" className="nav-cta" icon="shield" onClick={() => onNavigate('login')}>
            Secure Sign In
          </ButtonWithIcon>
        ) : (
          <ButtonWithIcon type="button" className="nav-cta" icon="logout" onClick={onLogout}>
            Sign Out
          </ButtonWithIcon>
        )}
      </aside>

      <main className="main-panel">
        <header className="topbar surface-glass">
          <button
            type="button"
            className="icon-button sidebar-toggle"
            onClick={onToggleSidebar}
            title={isSidebarCompact ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <Icon name="menu" />
          </button>
          <div className="search-shell">
            <Icon name="search" />
            <input placeholder="Search audits, candidates, reports" />
          </div>
          <div className="topbar-actions">
            {actions}
            {isAuthenticated && userProfile ? (
              <button type="button" className="profile-chip" onClick={() => onNavigate('settings')}>
                <span className="profile-avatar profile-avatar-small" aria-hidden="true">{userProfile.initials}</span>
                <span className="profile-chip-meta">
                  <strong>{userProfile.name}</strong>
                  <small>{userProfile.email}</small>
                </span>
                <Icon name="users" />
              </button>
            ) : null}
            <button type="button" className="icon-button" onClick={() => onNavigate('settings')}>
              <Icon name="settings" />
            </button>
          </div>
        </header>

        <div className="breadcrumb-trail surface-glass" aria-label="Breadcrumb">
          {crumbs.map((crumb, index) => (
            <React.Fragment key={crumb}>
              {index > 0 ? <span className="crumb-separator">/</span> : null}
              <span className={index === crumbs.length - 1 ? 'crumb-current' : ''}>{crumb}</span>
            </React.Fragment>
          ))}
        </div>

        {hasBackgroundLoad ? (
          <div className="top-progress" aria-label={loadLabel}>
            <span />
            <small>{loadLabel}</small>
          </div>
        ) : null}

        {isAuthenticated ? (
          <section className="guided-flow surface-card">
            {flowSteps.map(([label, done], index) => (
              <div key={label} className={done ? 'flow-step done' : 'flow-step'}>
                <span className="flow-index">{index + 1}</span>
                <span>{label}</span>
                <strong>{done ? '✓' : active === 'upload' && index === 0 ? '•' : '…'}</strong>
              </div>
            ))}
          </section>
        ) : null}

        <section className="page-content">{children}</section>

        <div className="mobile-nav surface-glass">
          {navItems.slice(0, 5).map(([route, , icon]) => (
            <button
              key={`mobile-${route}`}
              type="button"
              className={active === route ? 'mobile-nav-item active' : 'mobile-nav-item'}
              onClick={() => onNavigate(route)}
            >
              <Icon name={icon} />
            </button>
          ))}
        </div>
      </main>
    </div>
  )
}

function MetricCard({ label, value, note, accent = false, icon = 'spark' }) {
  return (
    <article className={accent ? 'metric-card metric-accent surface-card' : 'metric-card surface-card'}>
      <span className="eyebrow"><Icon name={icon} />{label}</span>
      <strong>{value}</strong>
      <p>{note}</p>
    </article>
  )
}

function SectionCard({ title, subtitle, children, className = '', icon = 'spark' }) {
  return (
    <article className={`section-card surface-card ${className}`.trim()}>
      {(title || subtitle) && (
        <header className="section-head">
          {subtitle && <span className="eyebrow"><Icon name={icon} />{subtitle}</span>}
          {title && <h2>{title}</h2>}
        </header>
      )}
      {children}
    </article>
  )
}

function SymbolFieldStrip({ items }) {
  return (
    <div className="symbol-field-strip">
      {items.map((item) => (
        <article key={item.title} className="symbol-field-chip surface-card">
          <span className="eyebrow"><Icon name={item.icon} />{item.label}</span>
          <strong>{item.title}</strong>
          <p>{item.text}</p>
        </article>
      ))}
    </div>
  )
}

async function callApi(path, options = {}) {
  if (!API_BASE) {
    throw new Error(API_CONFIG_ERROR)
  }

  const { token, headers = {}, ...requestOptions } = options
  const requestHeaders = { ...headers }
  if (token) {
    requestHeaders.Authorization = `Bearer ${token}`
  }

  let response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...requestOptions,
      headers: requestHeaders,
    })
  } catch (error) {
    throw new Error(`Cannot reach backend API at ${API_BASE}. ${IS_LOCAL_HOST ? 'Make sure FastAPI is running on port 8000.' : 'Deploy backend and set VITE_API_URL.'}`)
  }

  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json')
    ? await response.json().catch(() => ({}))
    : {}

  if (!contentType.includes('application/json')) {
    throw new Error(`Backend response was not JSON. Verify VITE_API_URL points to the API service, not the static frontend host.`)
  }

  if (!response.ok) {
    throw new Error(payload.detail || 'Request failed')
  }
  return payload
}

async function pollJobResult(jobId, token, onUpdate) {
  for (;;) {
    const status = await callApi(`/jobs/${encodeURIComponent(jobId)}`, { token })
    if (onUpdate) onUpdate(status)
    if (status.status === 'completed') {
      return status.result
    }
    if (status.status === 'failed') {
      throw new Error(status.error || status.message || 'Job failed')
    }
    await new Promise((resolve) => window.setTimeout(resolve, 700))
  }
}

function LandingPage({ onNavigate }) {
  const landingFields = [
    { icon: 'shield', label: 'Compliance', title: 'EEOC aligned checks', text: 'Every pipeline run keeps fairness metrics visible for audit teams.' },
    { icon: 'analysis', label: 'Explainability', title: 'Transparent scoring', text: 'Ranked feature influence helps reviewers justify hiring outcomes.' },
    { icon: 'reports', label: 'Readiness', title: 'Board-ready reports', text: 'Generate downloadable reports with model, bias, and rationale sections.' },
  ]

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
          <a href="#features"><LabelIcon icon="spark">Features</LabelIcon></a>
          <a href="#trust"><LabelIcon icon="shield">Trust</LabelIcon></a>
          <a href="#/login"><LabelIcon icon="login">Login</LabelIcon></a>
        </div>
      </header>

      <section className="hero-slab surface-panel">
        <div className="hero-copy">
          <span className="eyebrow"><Icon name="spark" />Architectural integrity for AI hiring</span>
          <h1>Design the hiring process around fairness, evidence, and calm authority.</h1>
          <p>
            FairHire AI turns candidate screening into a transparent audit flow with bias detection, explainability,
            and executive-ready reporting.
          </p>
          <div className="hero-actions">
            <ButtonWithIcon type="button" className="primary-button" icon="dashboard" onClick={() => onNavigate('dashboard')}>
              Enter Dashboard
            </ButtonWithIcon>
            <ButtonWithIcon type="button" className="secondary-button" icon="reports" onClick={() => onNavigate('reports')}>
              View Sample Audit
            </ButtonWithIcon>
          </div>
        </div>
        <div className="hero-visual surface-card">
          <div className="hero-band" />
          <div className="hero-band muted" />
          <div className="hero-band teal" />
          <div className="hero-chip"><Icon name="spark" />Real-time fairness instrumentation</div>
        </div>
      </section>

      <SymbolFieldStrip items={landingFields} />
    </div>
  )
}

function LoginPage({ onNavigate, onLogin, authLoading }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [errors, setErrors] = useState({})
  const loginFields = [
    { icon: 'shield', label: 'Required', title: 'Verified work email', text: 'Use your company domain email to access protected audit routes.' },
    { icon: 'login', label: 'Required', title: '8+ character password', text: 'Strong credentials keep sensitive candidate data secure.' },
    { icon: 'warning', label: 'Security', title: 'Session based access', text: 'Inactive sessions auto-expire to reduce unauthorized usage.' },
  ]

  const submit = (event) => {
    event.preventDefault()
    const nextErrors = {}
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      nextErrors.email = 'Enter a valid corporate email.'
    }
    if (password.length < 8) {
      nextErrors.password = 'Password must be at least 8 characters.'
    }
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length) return

    onLogin({ email, password })
  }

  return (
    <div className="login-page">
      <section className="login-panel login-hero surface-panel">
        <span className="eyebrow"><Icon name="shield" />Secure access</span>
        <h1>Enter the audit environment.</h1>
        <p>Use a clean, compliant login surface designed for enterprise review and high-trust team access.</p>
      </section>
      <section className="login-panel surface-card login-form-shell">
        <form className="login-form" onSubmit={submit}>
          <span className="eyebrow"><Icon name="login" />Sign in</span>
          <h2>Corporate access</h2>
          <label>
            Email
            <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="name@company.com" />
            {errors.email ? <small className="field-error">{errors.email}</small> : null}
          </label>
          <label>
            Password
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="••••••••" />
            {errors.password ? <small className="field-error">{errors.password}</small> : null}
          </label>
          <ButtonWithIcon type="submit" className="primary-button" icon="dashboard" disabled={authLoading}>
            {authLoading ? 'Signing in...' : 'Continue to Dashboard'}
          </ButtonWithIcon>
          <ButtonWithIcon type="button" className="secondary-button" icon="arrow-left" onClick={() => onNavigate('landing')}>
            Back to Landing
          </ButtonWithIcon>

          <SymbolFieldStrip items={loginFields} />
        </form>
      </section>
    </div>
  )
}

function DashboardPage({ onNavigate, biasData, trainData, loading }) {
  const biasChart = useMemo(() => {
    if (!biasData?.selection_rate_by_group) {
      return [
        { group: 'Female', value: 0.61 },
        { group: 'Male', value: 0.53 },
        { group: 'NonBinary', value: 0.58 },
      ]
    }
    return Object.entries(biasData.selection_rate_by_group).map(([group, value]) => ({ group, value }))
  }, [biasData])

  const dashboardFields = [
    { icon: 'warning', label: 'Monitor', title: 'Bias drift alerts', text: 'Watch fairness index shifts before they cross risk thresholds.' },
    { icon: 'users', label: 'Coverage', title: 'Group representation', text: 'Ensure all sensitive groups remain visible in selection analysis.' },
    { icon: 'spark', label: 'Action', title: 'Weekly review cadence', text: 'Schedule recurring checks to keep hiring models accountable.' },
  ]

  const fairnessIndex = biasData?.fairness_index ?? 0.84
  const parityDelta = biasData?.demographic_parity_difference ?? 0.14
  const verdict = fairnessIndex >= 0.85 ? 'Fair' : fairnessIndex >= 0.7 ? 'Moderate Risk' : 'High Risk'
  const verdictTone = verdict === 'Fair' ? 'green' : verdict === 'Moderate Risk' ? 'amber' : 'red'
  const verdictIcon = verdict === 'Fair' ? 'check' : 'warning'
  const verdictReason =
    verdict === 'Fair'
      ? `Fairness index is stable at ${fairnessIndex.toFixed(2)} with low observed drift.`
      : verdict === 'Moderate Risk'
        ? `Bias gap (Δ = ${parityDelta.toFixed(2)}) exceeds recommended threshold.`
        : `High risk detected: fairness index is ${fairnessIndex.toFixed(2)} and needs immediate intervention.`
  const verdictAction =
    verdict === 'Fair'
      ? 'Recommended action: Keep monitoring and export the current run.'
      : verdict === 'Moderate Risk'
        ? 'Recommended action: Adjust threshold or rebalance data.'
        : 'Recommended action: Pause deployment, rebalance data, and rerun the audit.'

  return (
    <>
      <section className={`verdict-banner ${verdictTone}`}>
        <div>
          <span className="eyebrow"><Icon name={verdictIcon} />Fairness Verdict</span>
          <h2>{verdict === 'Moderate Risk' ? 'FAIRNESS ALERT: Moderate Risk Detected' : `Status: ${verdict}`}</h2>
          <p>{verdictReason}</p>
          <p className="recommendation-line">{verdictAction}</p>
          <p className="micro-copy">Based on current model behavior.</p>
        </div>
        <span className={`status-chip ${verdictTone}`}>{verdict}</span>
      </section>

      <div className="metric-grid">
        <MetricCard label="Active audits" value="18" note="5 pending review" icon="reports" />
        <MetricCard label="Risk signal" value={(biasData?.fairness_index ?? 0.84).toFixed(2)} note="Teal threshold exceeded" accent icon="warning" />
        <MetricCard label="Accuracy" value={(trainData?.accuracy ?? 0.92).toFixed(2)} note="Latest trained model" icon="analysis" />
      </div>

      <div className="two-column-grid">
        <SectionCard title="Fairness distribution" subtitle="Selection rate parity across groups" icon="analysis">
          {loading.bias ? (
            <Skeleton className="chart-skeleton" />
          ) : (
            <>
              <div className="chart-host">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={biasChart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                    <XAxis dataKey="group" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]} fill="var(--chart-bar-primary)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="micro-copy">Derived from training dataset patterns.</p>
            </>
          )}
        </SectionCard>

        <SectionCard title="Key actions" subtitle="Operational shortcuts" icon="spark">
          <div className="action-stack">
            <ButtonWithIcon type="button" className="primary-button" icon="upload" onClick={() => onNavigate('upload')}>
              Upload new dataset
            </ButtonWithIcon>
            <ButtonWithIcon type="button" className="secondary-button" icon="bias" onClick={() => onNavigate('bias-report')}>
              Review bias findings
            </ButtonWithIcon>
            <ButtonWithIcon type="button" className="secondary-button" icon="reports" onClick={() => onNavigate('reports')}>
              Open report archive
            </ButtonWithIcon>
          </div>
        </SectionCard>
      </div>

      <SectionCard title="AI Audit Insights" subtitle="Model recommendations" icon="spark" className="ai-insights-card">
        <ul className="ai-list">
          <li><Icon name="analysis" />Model favors candidates with +2 yrs experience disproportionately.</li>
          <li><Icon name="warning" />Gender disparity detected (Δ = {parityDelta.toFixed(2)}).</li>
          <li><Icon name="bias" />Feature referral_source may introduce proxy bias.</li>
        </ul>
        <div className="recommend-grid">
          <span className="status-chip green"><Icon name="check" />Reweight dataset</span>
          <span className="status-chip green"><Icon name="check" />Remove sensitive proxy features</span>
          <span className="status-chip green"><Icon name="check" />Recalibrate threshold</span>
        </div>
      </SectionCard>

      <SymbolFieldStrip items={dashboardFields} />
    </>
  )
}

function UploadPage({
  onNavigate,
  loading,
  uploadData,
  onUpload,
  onTrain,
  selectedTarget,
  setSelectedTarget,
}) {
  const inputRef = useRef(null)
  const uploadFields = [
    { icon: 'file', label: 'Required', title: 'Target column mapping', text: 'Pick the decision label column before training begins.' },
    { icon: 'check', label: 'Quality', title: 'Schema consistency', text: 'Column names and data types should stay consistent across batches.' },
    { icon: 'shield', label: 'Privacy', title: 'Sensitive field tagging', text: 'Mark protected attributes to power fairness diagnostics.' },
  ]

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Upload status" value={loading.upload ? 'Uploading' : 'Ready'} note="Awaiting dataset selection" icon="upload" />
        <MetricCard label="Schema health" value={uploadData ? `${uploadData.columns.length}` : '0'} note="Detected columns" accent icon="check" />
        <MetricCard label="Rows" value={uploadData?.rows || 0} note="Current dataset size" icon="users" />
      </div>

      <div className="two-column-grid">
        <SectionCard title="Dataset intake" subtitle="Drag and drop or browse a file" icon="upload">
          <input
            ref={inputRef}
            className="hidden-input"
            type="file"
            accept=".csv,.json,.xlsx,.xls"
            onChange={(event) => {
              const selected = event.target.files?.[0]
              if (!selected) return
              onUpload(selected)
            }}
          />
          <div className="upload-dropzone" role="button" tabIndex={0} onClick={() => inputRef.current?.click()}>
            <Icon name="upload" />
            <strong>{loading.upload ? 'Uploading dataset...' : 'Browse files'}</strong>
            <p>CSV, JSON, or Excel datasets only.</p>
          </div>
        </SectionCard>

        <SectionCard title={uploadData?.filename || 'No dataset selected'} subtitle={uploadData ? `${uploadData.rows} rows` : 'Waiting for upload'} icon="file">
          {loading.upload ? (
            <Skeleton className="table-skeleton" />
          ) : uploadData ? (
            <>
              <label className="target-select-label">
                Target column
                <select value={selectedTarget} onChange={(event) => setSelectedTarget(event.target.value)}>
                  {(uploadData.target_suggestions || []).map((column) => (
                    <option key={column} value={column}>{column}</option>
                  ))}
                </select>
              </label>
              <div className="table-scroll">
                <table className="table-mock">
                  <thead>
                    <tr>{Object.keys(uploadData.preview?.[0] || {}).map((key) => <th key={key}>{key}</th>)}</tr>
                  </thead>
                  <tbody>
                    {(uploadData.preview || []).map((row, rowIndex) => (
                      <tr key={`row-${rowIndex}`}>
                        {Object.values(row).map((cell, cellIndex) => <td key={`cell-${cellIndex}`}>{String(cell)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="card-copy">Upload a dataset to preview schema and continue to training.</p>
          )}
        </SectionCard>
      </div>

      <div className="page-actions">
        <ButtonWithIcon type="button" className="secondary-button" icon="arrow-left" onClick={() => onNavigate('dashboard')}>
          Cancel
        </ButtonWithIcon>
        <ButtonWithIcon
          type="button"
          className="primary-button"
          icon="analysis"
          onClick={onTrain}
          disabled={!uploadData || loading.train}
        >
          {loading.train ? 'Training model...' : 'Continue to Mapping'}
        </ButtonWithIcon>
      </div>

      <SymbolFieldStrip items={uploadFields} />
    </>
  )
}

function ModelAnalysisPage({ onNavigate, trainData, loading }) {
  const confusionData = useMemo(() => {
    const matrix = trainData?.confusion_matrix || { tp: 141, fp: 19, tn: 128, fn: 15 }
    return [
      { name: 'TP', value: matrix.tp || 0 },
      { name: 'TN', value: matrix.tn || 0 },
      { name: 'FP', value: matrix.fp || 0 },
      { name: 'FN', value: matrix.fn || 0 },
    ]
  }, [trainData])

  const analysisFields = [
    { icon: 'analysis', label: 'Performance', title: 'Confusion balance', text: 'Compare false positives and false negatives before deployment.' },
    { icon: 'spark', label: 'Validation', title: 'Cross-check metrics', text: 'Use precision and recall together, not accuracy alone.' },
    { icon: 'bias', label: 'Next step', title: 'Fairness audit required', text: 'Run bias checks before approving candidate scoring in production.' },
  ]

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Accuracy" value={(trainData?.accuracy ?? 0.92).toFixed(2)} note="Balanced validation split" icon="analysis" />
        <MetricCard label="Precision" value={(trainData?.precision ?? 0.9).toFixed(2)} note="Prediction quality" accent icon="check" />
        <MetricCard label="Recall" value={(trainData?.recall ?? 0.89).toFixed(2)} note="Capture effectiveness" icon="spark" />
      </div>

      <div className="two-column-grid">
        <SectionCard title="Confusion matrix" subtitle="Model output distribution" icon="analysis">
          {loading.train ? (
            <Skeleton className="chart-skeleton" />
          ) : (
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={confusionData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" radius={[8, 8, 0, 0]} fill="var(--chart-bar-accent)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>
        <SectionCard title="Model insight" subtitle="Why the model behaves this way" icon="spark">
          <p className="card-copy">
            Training was completed on the currently uploaded dataset. Continue to bias auditing for subgroup fairness and
            then inspect explainability detail.
          </p>
          <p className="micro-copy">Based on current model behavior.</p>
          <ButtonWithIcon type="button" className="primary-button" icon="bias" onClick={() => onNavigate('bias-report')}>
            Run bias audit
          </ButtonWithIcon>
        </SectionCard>
      </div>

      <SectionCard title="Confidence layer" subtitle="Validation stability signal" icon="check">
        <div className="key-row">
          <span>Confidence</span>
          <strong>High</strong>
        </div>
        <p className="card-copy">Confidence is high based on validation stability across the current split.</p>
        <p className="micro-copy">Derived from training dataset patterns.</p>
      </SectionCard>

      <SymbolFieldStrip items={analysisFields} />
    </>
  )
}

function BiasReportPage({ onNavigate, biasData, loading, runId, sensitiveColumn, sensitiveOptions, onSensitiveColumnChange }) {
  const groupRows = Object.entries(biasData?.selection_rate_by_group || { Female: 0.61, Male: 0.53, NonBinary: 0.58 })
  const chartRows = groupRows.map(([group, value]) => ({ group, value: Number(value) }))
  const maxValue = Math.max(...chartRows.map((row) => row.value))
  const minValue = Math.min(...chartRows.map((row) => row.value))
  const groupGap = maxValue - minValue
  const [threshold, setThreshold] = useState(0.5)
  const [weightShift, setWeightShift] = useState(0)
  const baseFairness = biasData?.fairness_index ?? 0.68
  const adjustedFairness = Math.min(0.96, Math.max(0.5, baseFairness + (0.2 * (0.6 - Math.abs(weightShift - 0.2))) + (threshold - 0.5) * 0.12))
  const simulatedGap = Math.max(0.02, groupGap - (threshold - 0.5) * 0.08 - weightShift * 0.1)
  const biasFields = [
    { icon: 'warning', label: 'Required', title: 'Parity threshold review', text: 'Investigate groups where parity difference exceeds policy.' },
    { icon: 'users', label: 'Evidence', title: 'Selection by group', text: 'Track acceptance rate dispersion between demographic cohorts.' },
    { icon: 'settings', label: 'Mitigation', title: 'Threshold calibration', text: 'Tune decision limits and retrain to reduce disparity.' },
  ]

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Run" value={runId ? 'Active' : 'None'} note={runId || 'Upload and train first'} icon="reports" />
        <MetricCard
          label="Demographic parity"
          value={(biasData?.demographic_parity_difference ?? 0.14).toFixed(2)}
          note="Difference across groups"
          accent
          icon="warning"
        />
        <MetricCard
          label="Fairness index"
          value={(biasData?.fairness_index ?? 0.84).toFixed(2)}
          note="Closer to 1 is better"
          icon="check"
        />
      </div>

      <div className="two-column-grid">
        <SectionCard title="Selection rates" subtitle="By sensitive group" icon="bias">
          {loading.bias ? (
            <Skeleton className="table-skeleton" />
          ) : (
            <>
              <label className="target-select-label">
                Sensitive attribute
                <select
                  value={sensitiveColumn}
                  onChange={(event) => onSensitiveColumnChange(event.target.value)}
                  disabled={!sensitiveOptions?.length}
                >
                  {(sensitiveOptions || []).map((option) => (
                    <option key={option} value={option}>
                      {option.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                    </option>
                  ))}
                </select>
              </label>
              <div className="gap-headline">
                <strong>{(sensitiveColumn || 'group').replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())} Bias Gap: +{(groupGap * 100).toFixed(1)}%</strong>
                <span className={groupGap >= 0.08 ? 'status-chip amber' : 'status-chip green'}>{groupGap >= 0.08 ? 'Risk Indicator' : 'Within Safe Range'}</span>
              </div>
              <div className="chart-host">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartRows}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                    <XAxis dataKey="group" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <ReferenceLine y={0.6} stroke="var(--chart-threshold)" strokeDasharray="4 4" />
                    <ReferenceLine y={0.5} stroke="var(--chart-critical)" strokeDasharray="4 4" />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {chartRows.map((entry) => (
                        <Cell key={entry.group} fill={entry.value < 0.5 ? 'var(--chart-critical)' : entry.value < 0.6 ? 'var(--chart-threshold)' : 'var(--chart-bar-primary)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="risk-zone-legend">
                <span><i className="zone-critical" />Critical zone (&lt; 0.50)</span>
                <span><i className="zone-caution" />Caution zone (0.50 - 0.60)</span>
              </div>
              <p className="micro-copy">Sensitive attribute impact detected.</p>
              <div className="stacked-copy">
                {groupRows.map(([group, value]) => (
                  <div key={group} className="key-row">
                    <span>{group}</span>
                    <strong>{(Number(value) * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
            </>
          )}
        </SectionCard>
        <SectionCard title="Controls" subtitle="Next steps" icon="settings">
          <div className="action-stack">
            <ButtonWithIcon type="button" className="primary-button" icon="download" onClick={() => onNavigate('reports')}>
              Export report
            </ButtonWithIcon>
            <ButtonWithIcon type="button" className="secondary-button" icon="settings" onClick={() => onNavigate('settings')}>
              Adjust thresholds
            </ButtonWithIcon>
          </div>

          <section className="simulator-box">
            <h3><Icon name="spark" />What-if Simulator</h3>
            <label>
              Threshold: {threshold.toFixed(2)}
              <input type="range" min="0.35" max="0.75" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            </label>
            <label>
              Reweight strength: {weightShift.toFixed(2)}
              <input type="range" min="0" max="1" step="0.01" value={weightShift} onChange={(event) => setWeightShift(Number(event.target.value))} />
            </label>
            <div className="sim-output">
              <div className="key-row"><span>Simulated fairness index</span><strong>{adjustedFairness.toFixed(2)}</strong></div>
              <div className="key-row"><span>Simulated bias gap</span><strong>{(simulatedGap * 100).toFixed(1)}%</strong></div>
            </div>
          </section>
        </SectionCard>
      </div>

      <SectionCard title="Before vs After Fairness" subtitle="Mitigation impact snapshot" icon="check">
        <div className="before-after-grid">
          <article className="surface-card before-block">
            <span className="eyebrow"><Icon name="warning" />Before mitigation</span>
            <strong>Fairness Index: 0.68</strong>
            <p>High disparity risk with sensitive group imbalance.</p>
          </article>
          <article className="surface-card after-block">
            <span className="eyebrow"><Icon name="check" />After mitigation</span>
            <strong>Fairness Index: 0.84</strong>
            <p>Gap reduced and parity improved after threshold and reweighting updates.</p>
          </article>
        </div>
        <p className="micro-copy">Based on current model behavior.</p>
      </SectionCard>

      <SymbolFieldStrip items={biasFields} />
    </>
  )
}

function ExplainabilityPage({ onNavigate, explainData, loading }) {
  const featureData = explainData?.top_global_features || [
    { feature: 'experience', importance: 0.31 },
    { feature: 'referral_source', importance: 0.24 },
    { feature: 'assessment_score', importance: 0.19 },
  ]

  const explainFields = [
    { icon: 'explain', label: 'Required', title: 'Feature traceability', text: 'Explain top predictors used in each hiring recommendation.' },
    { icon: 'file', label: 'Documentation', title: 'Decision rationale', text: 'Store concise explanations for adverse and accepted outcomes.' },
    { icon: 'download', label: 'Audit', title: 'Export explain logs', text: 'Attach explainability evidence to compliance reports.' },
  ]

  const candidate = {
    id: 1023,
    decision: 'Rejected',
    factors: [
      { name: 'Experience', impact: '-12%' },
      { name: 'Education Tier', impact: '+8%' },
      { name: 'Referral Source', impact: '-6%' },
    ],
    explanation: 'Candidate lacked minimum experience threshold and model weighted it heavily.',
  }

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Sample size" value={explainData?.sample_size || 40} note="Records explained" icon="file" />
        <MetricCard label="Top feature" value={featureData[0]?.feature || 'experience'} note="Highest influence" accent icon="spark" />
        <MetricCard label="Signals" value={featureData.length} note="Ranked contributors" icon="analysis" />
      </div>

      <div className="two-column-grid">
        <SectionCard title="Global feature influence" subtitle="Top weighted variables" icon="analysis">
          {loading.explain ? (
            <Skeleton className="chart-skeleton" />
          ) : (
            <>
              <div className="chart-host">
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={featureData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
                    <XAxis type="number" domain={[0, 'dataMax']} />
                    <YAxis type="category" dataKey="feature" width={100} />
                    <Tooltip />
                    <Bar dataKey="importance" radius={[0, 8, 8, 0]} fill="var(--chart-bar-primary)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="micro-copy">Derived from training dataset patterns.</p>
            </>
          )}
        </SectionCard>
        <SectionCard title="Actions" subtitle="Compliance outputs" icon="download">
          <div className="action-stack">
            <ButtonWithIcon type="button" className="secondary-button" icon="download" onClick={() => onNavigate('reports')}>
              Download audit
            </ButtonWithIcon>
            <ButtonWithIcon type="button" className="primary-button" icon="reports" onClick={() => onNavigate('reports')}>
              Open report archive
            </ButtonWithIcon>
          </div>
        </SectionCard>
      </div>

      <SectionCard title="Candidate Insight" subtitle="Why this candidate got rejected" icon="users">
        <div className="candidate-box">
          <div className="key-row"><span>Candidate ID</span><strong>{candidate.id}</strong></div>
          <div className="key-row"><span>Decision</span><strong>❌ {candidate.decision}</strong></div>
          <div className="candidate-summary">Candidate lacked required experience threshold and the model weighted it heavily.</div>
          <div className="stacked-copy">
            <strong>Top contributing factors:</strong>
            {candidate.factors.map((factor) => (
              <div key={factor.name} className="key-row">
                <span>{factor.name}</span>
                <strong>{factor.impact}</strong>
              </div>
            ))}
          </div>
          <p className="card-copy"><strong>Explanation:</strong> {candidate.explanation}</p>
          <p className="micro-copy">Based on current model behavior.</p>
        </div>
      </SectionCard>

      <SymbolFieldStrip items={explainFields} />
    </>
  )
}

function ReportsPage({ reportData, loading }) {
  const reportSummary = useMemo(() => {
    const verified = reportData ? 98 : 92
    const pending = reportData ? 16 : 22
    return [
      { name: 'Verified', value: verified, color: '#14b8a6' },
      { name: 'Pending', value: pending, color: '#f59e0b' },
    ]
  }, [reportData])

  const reportFields = [
    { icon: 'reports', label: 'Required', title: 'Versioned reports', text: 'Keep immutable snapshots for every model release cycle.' },
    { icon: 'check', label: 'Governance', title: 'Reviewer sign-off', text: 'Capture approver names and decision timestamps.' },
    { icon: 'download', label: 'Distribution', title: 'Export package', text: 'Share report bundles with legal and HR leadership.' },
  ]

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Stored reports" value="128" note="Connected to API" icon="reports" />
        <MetricCard label="Verified" value="98" note="Cleanly signed off" accent icon="check" />
        <MetricCard label="Pending" value="16" note="Requires human review" icon="warning" />
      </div>

      <div className="two-column-grid">
        <SectionCard title="Audit status" subtitle="Verification distribution" icon="reports">
          {loading.report ? (
            <Skeleton className="chart-skeleton" />
          ) : (
            <div className="chart-host">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={reportSummary} dataKey="value" innerRadius={50} outerRadius={80} paddingAngle={3}>
                    {reportSummary.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>

        <SectionCard title="Latest report" subtitle="Current run summary" icon="file">
          <div className="stacked-copy">
            <div className="key-row"><span>Model accuracy</span><strong>{(reportData?.train?.accuracy ?? 0.92).toFixed(2)}</strong></div>
            <div className="key-row"><span>Fairness index</span><strong>{(reportData?.bias?.fairness_index ?? 0.84).toFixed(2)}</strong></div>
            <div className="key-row"><span>Top feature</span><strong>{reportData?.explain?.top_global_features?.[0]?.feature || 'experience'}</strong></div>
          </div>
        </SectionCard>
      </div>

      <SectionCard title="Board-ready report preview" subtitle="Executive compliance format" icon="reports">
        <div className="executive-grid">
          <article className="surface-card executive-block">
            <h3>Summary</h3>
            <p>Current model performance remains high with moderate fairness risk requiring follow-up.</p>
          </article>
          <article className="surface-card executive-block">
            <h3>Risk Areas</h3>
            <p>Gender parity gap and referral source influence require immediate policy review.</p>
          </article>
          <article className="surface-card executive-block">
            <h3>Compliance Status</h3>
            <p>Provisionally compliant pending threshold recalibration and secondary validation run.</p>
          </article>
          <article className="surface-card executive-block">
            <h3>Recommendations</h3>
            <p>Reweight dataset, rerun fairness test, and attach updated explainability appendix.</p>
          </article>
        </div>
        <p className="micro-copy">Derived from training dataset patterns.</p>
      </SectionCard>

      <SymbolFieldStrip items={reportFields} />
    </>
  )
}

function SettingsPage({ themeMode, onThemeModeChange, effectiveTheme, user }) {
  const [tealAlerts, setTealAlerts] = useState(true)
  const [explainability, setExplainability] = useState(false)
  const [saved, setSaved] = useState(false)
  const [activeSection, setActiveSection] = useState('general')

  const menuItems = [
    { key: 'general', label: 'General', icon: 'settings' },
    { key: 'model-controls', label: 'Model controls', icon: 'analysis' },
    { key: 'security', label: 'Security', icon: 'shield' },
    { key: 'retention', label: 'Retention', icon: 'reports' },
  ]

  const renderSectionContent = () => {
    if (activeSection === 'general') {
      return (
        <div className="settings-stack">
          <SectionCard title="Platform preferences" subtitle="Workspace defaults" icon="settings">
            <div className="theme-mode-group" role="radiogroup" aria-label="Theme mode">
              <button
                type="button"
                className={themeMode === 'light' ? 'theme-mode-button active' : 'theme-mode-button'}
                onClick={() => onThemeModeChange('light')}
                role="radio"
                aria-checked={themeMode === 'light'}
              >
                Light
              </button>
              <button
                type="button"
                className={themeMode === 'dark' ? 'theme-mode-button active' : 'theme-mode-button'}
                onClick={() => onThemeModeChange('dark')}
                role="radio"
                aria-checked={themeMode === 'dark'}
              >
                Dark
              </button>
              <button
                type="button"
                className={themeMode === 'device' ? 'theme-mode-button active' : 'theme-mode-button'}
                onClick={() => onThemeModeChange('device')}
                role="radio"
                aria-checked={themeMode === 'device'}
              >
                Device
              </button>
            </div>
            <div className="stacked-copy">
              <div className="key-row"><span>Interface mode</span><strong>{themeMode === 'device' ? 'Device' : themeMode === 'dark' ? 'Dark' : 'Light'}</strong></div>
              <div className="key-row"><span>Applied theme</span><strong>{effectiveTheme === 'dark' ? 'Dark' : 'Light'}</strong></div>
              <div className="key-row"><span>Chart density</span><strong>Balanced</strong></div>
              <div className="key-row"><span>Toast style</span><strong>Insight mode</strong></div>
            </div>
          </SectionCard>

          <SectionCard title="Profile details" subtitle="Registered user identity" icon="users">
            <div className="stacked-copy">
              <div className="key-row"><span>User name</span><strong>{user?.name || 'FairHire User'}</strong></div>
              <div className="key-row"><span>Email</span><strong>{user?.email || 'Not available'}</strong></div>
              <div className="key-row"><span>User ID</span><strong>{user?.user_id || 'Not assigned yet'}</strong></div>
              <div className="key-row"><span>Employee ID</span><strong>{user?.employee_id || 'Not assigned yet'}</strong></div>
              <div className="key-row"><span>Role</span><strong>{user?.role || 'analyst'}</strong></div>
            </div>
          </SectionCard>
        </div>
      )
    }

    if (activeSection === 'model-controls') {
      return (
        <div className="settings-stack">
          <SectionCard title="Model configuration" subtitle="Primary thresholds and AI behavior" icon="settings">
            <div className="setting-row">
              <div>
                <strong><Icon name="warning" />Auto-flag protected classes</strong>
                <p>Highlight sensitive fields during ingest.</p>
              </div>
              <button type="button" className={tealAlerts ? 'toggle on' : 'toggle'} onClick={() => setTealAlerts((value) => !value)}>
                <span />
              </button>
            </div>
            <div className="setting-row">
              <div>
                <strong><Icon name="file" />Enforce explainability bundle</strong>
                <p>Require a candidate-level rationale for adverse actions.</p>
              </div>
              <button type="button" className={explainability ? 'toggle on' : 'toggle'} onClick={() => setExplainability((value) => !value)}>
                <span />
              </button>
            </div>
          </SectionCard>
        </div>
      )
    }

    if (activeSection === 'security') {
      return (
        <div className="settings-stack">
          <SectionCard title="Security policies" subtitle="Access and release controls" icon="shield">
            <div className="stacked-copy">
              <div className="key-row"><span>Session timeout</span><strong>30 minutes</strong></div>
              <div className="key-row"><span>MFA requirement</span><strong>Enabled</strong></div>
              <div className="key-row"><span>Admin override log</span><strong>Enabled</strong></div>
            </div>
          </SectionCard>
        </div>
      )
    }

    return (
      <div className="settings-stack">
        <SectionCard title="Retention policy" subtitle="Storage and compliance windows" icon="reports">
          <div className="stacked-copy">
            <div className="key-row"><span>Data retention</span><strong>24 months</strong></div>
            <div className="key-row"><span>Audit export archive</span><strong>Enabled</strong></div>
            <div className="key-row"><span>Purge approval</span><strong>Required</strong></div>
          </div>
        </SectionCard>
      </div>
    )
  }
  const settingFields = [
    { icon: 'shield', label: 'Required', title: 'Security baseline', text: 'Keep session timeout and access controls reviewed monthly.' },
    { icon: 'analysis', label: 'Model Ops', title: 'Drift watchlist', text: 'Define who gets alerts when fairness quality drops.' },
    { icon: 'reports', label: 'Retention', title: 'Policy archive', text: 'Preserve audit artifacts for the required legal window.' },
  ]

  return (
    <>
      <div className="settings-grid">
        <aside className="surface-card settings-menu">
          {menuItems.map((item) => (
            <button
              key={item.key}
              type="button"
              className={activeSection === item.key ? 'active' : ''}
              onClick={() => setActiveSection(item.key)}
            >
              <Icon name={item.icon} />
              {item.label}
            </button>
          ))}
        </aside>

        {renderSectionContent()}
      </div>

      <div className="page-actions">
        <ButtonWithIcon type="button" className="secondary-button" icon="warning" onClick={() => setSaved(false)}>
          Discard
        </ButtonWithIcon>
        <ButtonWithIcon type="button" className="primary-button" icon="check" onClick={() => setSaved(true)}>
          Save changes
        </ButtonWithIcon>
      </div>
      {saved && <span className="status-chip green"><Icon name="check" />Saved</span>}

      <SymbolFieldStrip items={settingFields} />
    </>
  )
}

function useToasts() {
  const [toasts, setToasts] = useState([])

  const pushToast = (type, title, message, insight = '') => {
    const id = `toast_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`
    setToasts((current) => [...current, { id, type, title, message, insight }])
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id))
    }, 4200)
  }

  return {
    toasts,
    pushToast,
    dismissToast: (id) => setToasts((current) => current.filter((toast) => toast.id !== id)),
  }
}

export default function App() {
  const [route, setRoute] = useState(readRoute)
  const [session, setSession] = useState(() => {
    try {
      const raw = localStorage.getItem(SESSION_KEY)
      const parsed = raw ? JSON.parse(raw) : null
      if (parsed?.token && parsed?.user) return parsed
      if (parsed?.token && parsed?.email) {
        return { token: parsed.token, user: { email: parsed.email, name: deriveDisplayName(parsed.email), role: 'analyst' } }
      }
      return null
    } catch {
      return null
    }
  })

  const [uploadData, setUploadData] = useState(null)
  const [selectedTarget, setSelectedTarget] = useState('')
  const [trainData, setTrainData] = useState(null)
  const [biasData, setBiasData] = useState(null)
  const [explainData, setExplainData] = useState(null)
  const [reportData, setReportData] = useState(null)
  const [sensitiveColumn, setSensitiveColumn] = useState('gender')
  const [biasError, setBiasError] = useState(null)
  const [explainError, setExplainError] = useState(null)
  const [reportError, setReportError] = useState(null)
  const [isSidebarCompact, setIsSidebarCompact] = useState(false)
  const [uiBooting, setUiBooting] = useState(true)
  const [routeStageClass, setRouteStageClass] = useState('entered')
  const [themeMode, setThemeMode] = useState(() => {
    try {
      const saved = localStorage.getItem(THEME_KEY)
      return saved === 'light' || saved === 'dark' || saved === 'device' ? saved : 'device'
    } catch {
      return 'device'
    }
  })
  const [systemPrefersDark, setSystemPrefersDark] = useState(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  const sensitiveOptions = useMemo(() => {
    const cols = uploadData?.columns || []
    const target = selectedTarget || trainData?.target_column
    const filtered = cols.filter((col) => col !== target)
    if (biasData?.sensitive_column && !filtered.includes(biasData.sensitive_column)) {
      return [biasData.sensitive_column, ...filtered]
    }
    return filtered
  }, [uploadData, selectedTarget, trainData?.target_column, biasData?.sensitive_column])

  const [loading, setLoading] = useState({
    auth: false,
    upload: false,
    train: false,
    bias: false,
    explain: false,
    report: false,
  })

  const { toasts, pushToast, dismissToast } = useToasts()

  const isAuthenticated = Boolean(session?.token)
  const runId = trainData?.run_id || null
  const effectiveTheme = themeMode === 'device' ? (systemPrefersDark ? 'dark' : 'light') : themeMode
  const userProfile = useMemo(() => {
    const email = session?.user?.email
    if (!email) return null
    const name = session?.user?.name || deriveDisplayName(email)
    return {
      email,
      name,
      initials: deriveInitials(name),
    }
  }, [session])

  useEffect(() => {
    localStorage.setItem(SESSION_KEY, JSON.stringify(session || null))
  }, [session])

  useEffect(() => {
    localStorage.setItem(THEME_KEY, themeMode)
  }, [themeMode])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', effectiveTheme)
  }, [effectiveTheme])

  useEffect(() => {
    const timer = window.setTimeout(() => setUiBooting(false), 520)
    return () => window.clearTimeout(timer)
  }, [])

  useEffect(() => {
    const onHashChange = () => setRoute(readRoute())
    window.addEventListener('hashchange', onHashChange)
    if (!window.location.hash) {
      navigate('landing')
    }
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const apply = () => setSystemPrefersDark(media.matches)
    apply()
    media.addEventListener('change', apply)
    return () => media.removeEventListener('change', apply)
  }, [])

  useEffect(() => {
    setRouteStageClass('entering')
    const timer = window.setTimeout(() => setRouteStageClass('entered'), 260)
    return () => window.clearTimeout(timer)
  }, [route])

  useEffect(() => {
    if (PROTECTED_ROUTES.has(route) && !isAuthenticated) {
      navigate('login')
      setRoute('login')
    }
  }, [route, isAuthenticated])

  useEffect(() => {
    const media = window.matchMedia('(max-width: 1360px)')
    const apply = () => setIsSidebarCompact(media.matches)
    apply()
    media.addEventListener('change', apply)
    return () => media.removeEventListener('change', apply)
  }, [])

  useEffect(() => {
    if (!sensitiveOptions.length) return
    if (!sensitiveOptions.includes(sensitiveColumn)) {
      setSensitiveColumn(sensitiveOptions[0])
      setBiasData(null)
      setBiasError(null)
    }
  }, [sensitiveOptions, sensitiveColumn])

  useEffect(() => {
    const loadBias = async () => {
      if (!runId || route !== 'bias-report' || biasData || loading.bias || biasError) return
      setLoading((prev) => ({ ...prev, bias: true }))
      try {
        const payload = await callApi(`/bias?run_id=${encodeURIComponent(runId)}&sensitive_column=${encodeURIComponent(sensitiveColumn)}`)
        setBiasData(payload)
        setBiasError(null)
        pushToast(
          'success',
          'Bias audit complete',
          `Audited ${payload.sensitive_column.replace(/_/g, ' ')} across group parity.`,
          `Insight: Fairness index ${(payload.fairness_index * 100).toFixed(1)}% for this run.`,
        )
      } catch (error) {
        setBiasError(error.message || 'Bias analysis failed')
        pushToast('error', 'Bias analysis failed', error.message, 'Insight: Select a sensitive attribute present in the uploaded dataset schema.')
      } finally {
        setLoading((prev) => ({ ...prev, bias: false }))
      }
    }

    loadBias()
  }, [runId, route, biasData, loading.bias, sensitiveColumn, biasError])

  useEffect(() => {
    const loadExplain = async () => {
      if (!runId || route !== 'explainability' || explainData || loading.explain || explainError) return
      setLoading((prev) => ({ ...prev, explain: true }))
      try {
        const submission = await callApi(`/explain?run_id=${encodeURIComponent(runId)}`, { token: session?.token })
        const payload = submission.result || await pollJobResult(submission.job_id, session?.token, (status) => {
          pushToast('info', `Explain ${status.status}`, status.message)
        })
        setExplainData(payload)
        setExplainError(null)
      } catch (error) {
        setExplainError(error.message || 'Explainability failed')
        pushToast('error', 'Explainability failed', error.message)
      } finally {
        setLoading((prev) => ({ ...prev, explain: false }))
      }
    }

    loadExplain()
  }, [runId, route, explainData, loading.explain, session?.token, explainError])

  useEffect(() => {
    const loadReport = async () => {
      if (!runId || route !== 'reports' || reportData || loading.report || reportError) return
      setLoading((prev) => ({ ...prev, report: true }))
      try {
        const payload = await callApi(`/report?run_id=${encodeURIComponent(runId)}`, { token: session?.token })
        setReportData(payload)
        setReportError(null)
        saveGeneratedReport({ user: session?.user, report: payload }).catch(() => {
          pushToast('info', 'Firestore sync', 'Report generated, but Firestore sync failed.')
        })
      } catch (error) {
        setReportError(error.message || 'Report generation failed')
        pushToast('error', 'Report generation failed', error.message)
      } finally {
        setLoading((prev) => ({ ...prev, report: false }))
      }
    }

    loadReport()
  }, [runId, route, reportData, loading.report, session?.token, reportError])

  const handleLogin = async ({ email, password }) => {
    setLoading((prev) => ({ ...prev, auth: true }))
    try {
      let authPayload
      try {
        authPayload = await callApi('/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password }),
        })
      } catch (error) {
        if (String(error.message || '').includes('Invalid email or password')) {
          authPayload = await callApi('/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
          })
        } else {
          throw error
        }
      }

      setSession(authPayload)
      upsertUserProfile(authPayload.user).catch(() => {
        pushToast('info', 'Firestore sync', 'Login succeeded, but profile sync to Firestore failed.')
      })
      pushToast('success', 'Signed in', `Welcome back, ${authPayload.user.name}.`)
      navigate('dashboard')
      setRoute('dashboard')
    } catch (error) {
      pushToast('error', 'Sign in failed', error.message || 'Unable to sign in right now.')
    } finally {
      setLoading((prev) => ({ ...prev, auth: false }))
    }
  }

  const handleLogout = () => {
    setSession(null)
    setUploadData(null)
    setTrainData(null)
    setBiasData(null)
    setExplainData(null)
    setReportData(null)
    setSensitiveColumn('gender')
    setBiasError(null)
    setExplainError(null)
    setReportError(null)
    pushToast('info', 'Signed out', 'Session cleared from this browser.')
    navigate('landing')
    setRoute('landing')
  }

  const handleUpload = async (file) => {
    setLoading((prev) => ({ ...prev, upload: true }))
    setBiasData(null)
    setExplainData(null)
    setReportData(null)
    setTrainData(null)
    setSensitiveColumn('gender')
    setBiasError(null)
    setExplainError(null)
    setReportError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)

      let payload
      payload = await callApi('/upload', { method: 'POST', body: formData })

      setUploadData(payload)
      setSelectedTarget(payload.target_suggestions?.[0] || payload.columns?.[0] || '')
      saveDatasetUpload({
        user: session?.user,
        upload: payload,
        selectedTarget: payload.target_suggestions?.[0] || payload.columns?.[0] || null,
      }).catch(() => {
        pushToast('info', 'Firestore sync', 'Dataset uploaded, but Firestore sync failed.')
      })
      pushToast('success', 'Dataset ready', `Loaded ${payload.rows} records from ${payload.filename}.`)
    } catch (error) {
      pushToast('error', 'Upload failed', error.message)
    } finally {
      setLoading((prev) => ({ ...prev, upload: false }))
    }
  }

  const handleTrain = async () => {
    if (!uploadData?.dataset_id) {
      pushToast('error', 'No dataset', 'Upload a dataset before training.')
      return
    }

    setLoading((prev) => ({ ...prev, train: true }))
    try {
      let payload
      const target = selectedTarget || uploadData.target_suggestions?.[0] || uploadData.columns?.[0]
      const submission = await callApi('/train', {
        method: 'POST',
        token: session?.token,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: uploadData.dataset_id,
          target_column: target,
          model_type: 'random_forest',
        }),
      })
      payload = submission.result || await pollJobResult(submission.job_id, session?.token, (status) => {
        pushToast('info', `Training ${status.status}`, status.message)
      })

      setTrainData(payload)
      setBiasError(null)
      setExplainError(null)
      setReportError(null)
      saveTrainingRun({ user: session?.user, training: payload }).catch(() => {
        pushToast('info', 'Firestore sync', 'Training completed, but run sync to Firestore failed.')
      })
      pushToast('success', 'Model trained', `Run ${payload.run_id} is ready for bias analysis.`, `Insight: Move to Fairness Audit to compare group selection rates.`)
      navigate('model-analysis')
      setRoute('model-analysis')
    } catch (error) {
      pushToast('error', 'Training failed', error.message)
    } finally {
      setLoading((prev) => ({ ...prev, train: false }))
    }
  }

  const wrappedPage = useMemo(() => {
    if (PROTECTED_ROUTES.has(route) && !isAuthenticated) {
      return <LoginPage onNavigate={navigate} onLogin={handleLogin} authLoading={loading.auth} />
    }

    switch (route) {
      case 'login':
        return <LoginPage onNavigate={navigate} onLogin={handleLogin} authLoading={loading.auth} />
      case 'dashboard':
        return (
          <AppShell active="dashboard" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <DashboardPage onNavigate={navigate} biasData={biasData} trainData={trainData} loading={loading} />
          </AppShell>
        )
      case 'upload':
        return (
          <AppShell active="upload" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <UploadPage
              onNavigate={navigate}
              loading={loading}
              uploadData={uploadData}
              onUpload={handleUpload}
              onTrain={handleTrain}
              selectedTarget={selectedTarget}
              setSelectedTarget={setSelectedTarget}
            />
          </AppShell>
        )
      case 'model-analysis':
        return (
          <AppShell active="model-analysis" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <ModelAnalysisPage onNavigate={navigate} trainData={trainData} loading={loading} />
          </AppShell>
        )
      case 'bias-report':
        return (
          <AppShell active="bias-report" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <BiasReportPage
              onNavigate={navigate}
              biasData={biasData}
              loading={loading}
              runId={runId}
              sensitiveColumn={sensitiveColumn}
              sensitiveOptions={sensitiveOptions}
              onSensitiveColumnChange={(column) => {
                setSensitiveColumn(column)
                setBiasData(null)
                setBiasError(null)
              }}
            />
          </AppShell>
        )
      case 'explainability':
        return (
          <AppShell active="explainability" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <ExplainabilityPage onNavigate={navigate} explainData={explainData} loading={loading} />
          </AppShell>
        )
      case 'reports':
        return (
          <AppShell active="reports" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <ReportsPage reportData={reportData} loading={loading} />
          </AppShell>
        )
      case 'settings':
        return (
          <AppShell active="settings" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <SettingsPage
              themeMode={themeMode}
              onThemeModeChange={setThemeMode}
              effectiveTheme={effectiveTheme}
              user={session?.user}
            />
          </AppShell>
        )
      case 'landing':
      default:
        return <LandingPage onNavigate={navigate} />
    }
  }, [
    route,
    isAuthenticated,
    loading,
    biasData,
    trainData,
    uploadData,
    selectedTarget,
    explainData,
    reportData,
    runId,
    sensitiveColumn,
    sensitiveOptions,
    themeMode,
    effectiveTheme,
  ])

  return (
    <ErrorBoundary>
      <GlobalLoadingOverlay
        visible={uiBooting || loading.auth}
        label={loading.auth ? 'Authenticating secure session' : 'Launching FairHire AI'}
      />
      <div className={`route-stage ${routeStageClass}`}>
        {wrappedPage}
      </div>
      <ToastStack toasts={toasts} onDismiss={dismissToast} />
    </ErrorBoundary>
  )
}
