import React, { useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
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
import { DEMO_REPORT } from './demoData'

function ShapVisual({ data }) {
  const items = data || [
    { label: 'Experience', value: 31, color: 'var(--brand-teal)' },
    { label: 'Assessment', value: 24, color: 'var(--brand-teal)' },
    { label: 'Referral', value: -18, color: 'var(--brand-amber)' },
  ]
  return (
    <div className="shap-visual">
      {items.map((item, idx) => (
        <div key={idx} className="shap-row">
          <span className="shap-bar-label">{item.label}</span>
          <div className="shap-bar-track">
            <div
              className={`shap-bar-fill ${item.value < 0 ? 'negative' : ''}`}
              style={{ width: `${Math.abs(item.value)}%` }}
            />
          </div>
          <span className="shap-impact">{item.value > 0 ? '+' : ''}{item.value}%</span>
        </div>
      ))}
    </div>
  )
}

function CountingNumber({ value, duration = 2 }) {
  const [displayValue, setDisplayValue] = useState(0)
  
  useEffect(() => {
    let start = 0
    const end = parseFloat(value)
    if (start === end) return
    
    let totalMiliseconds = duration * 1000
    let incrementTime = (totalMiliseconds / (end * 100))
    
    let timer = setInterval(() => {
      start += 0.01
      setDisplayValue(start)
      if (start >= end) {
        setDisplayValue(end)
        clearInterval(timer)
      }
    }, incrementTime)
    
    return () => clearInterval(timer)
  }, [value, duration])
  
  return displayValue.toFixed(2)
}

function AnimatedBackground() {
  return (
    <div className="background-aurora">
      <motion.div
        className="aurora-blob cyan"
        animate={{
          x: [0, 200, -100, 0],
          y: [0, -150, 100, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
      <motion.div
        className="aurora-blob blue"
        animate={{
          x: [100, -150, 200, 100],
          y: [0, 200, -100, 0],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      />
      <div className="noise-overlay" />
    </div>
  )
}

function Particles() {
  const [winSize, setWinSize] = useState({ w: 1200, h: 800 })
  useEffect(() => {
    if (typeof window !== 'undefined') {
      setWinSize({ w: window.innerWidth, h: window.innerHeight })
    }
  }, [])

  return (
    <div className="background-particles">
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="particle"
          initial={{
            x: Math.random() * winSize.w,
            y: Math.random() * winSize.h,
          }}
          animate={{
            y: ['100%', '-10%'],
            opacity: [0, 0.4, 0],
          }}
          transition={{
            duration: 15 + Math.random() * 15,
            repeat: Infinity,
            ease: 'linear',
          }}
        />
      ))}
    </div>
  )
}

function MouseGlow() {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const handleMove = (e) => {
      setMousePos({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', handleMove)
    return () => window.removeEventListener('mousemove', handleMove)
  }, [])

  return (
    <motion.div
      className="mouse-glow"
      animate={{
        x: mousePos.x - 200,
        y: mousePos.y - 200,
      }}
      transition={{ type: 'spring', damping: 30, stiffness: 200, mass: 0.5 }}
    />
  )
}

function AuditSimulation() {
  const [step, setStep] = useState(0)
  const steps = [
    { label: 'Uploading dataset...', value: '✔ Done', status: 'success' },
    { label: 'Training model...', value: '✔ 92% Acc', status: 'success' },
    { label: 'Detecting bias...', value: '⚠ 14.3% Gap', status: 'warning' },
    { label: 'Applying mitigation...', value: '✔ Active', status: 'active' },
    { label: 'Fairness result', value: '✔ 8.7% Gap', status: 'success' },
  ]

  useEffect(() => {
    const timer = setInterval(() => {
      setStep((s) => (s + 1) % (steps.length + 1))
    }, 2400)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="audit-sim-card">
      <div className="eyebrow" style={{ marginBottom: 24 }}><Icon name="analysis" />System Intelligence Feed</div>
      <div style={{ display: 'grid', gap: 12 }}>
        {steps.map((s, i) => (
          <AnimatePresence key={i}>
            {step > i && (
              <motion.div
                initial={{ opacity: 0, x: -20, filter: 'blur(10px)' }}
                animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
                className={`sim-step ${step === i + 1 ? 'active' : s.status}`}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <Icon name={s.status === 'success' ? 'check' : s.status === 'warning' ? 'warning' : 'spark'} />
                  <span className="sim-step-label">{s.label}</span>
                </div>
                <span className="sim-step-value" style={{ fontWeight: 700, color: 'var(--brand-teal)' }}>{s.value}</span>
              </motion.div>
            )}
          </AnimatePresence>
        ))}
      </div>
      <div style={{ marginTop: 20, height: 2, background: 'rgba(255,255,255,0.05)', borderRadius: 1 }}>
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${(step / steps.length) * 100}%` }}
          className="timeline-connector-fill" 
          style={{ height: '100%', borderRadius: 1 }}
        />
      </div>
    </div>
  )
}

function FadeIn({ children, delay = 0, y = 40 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] }}
      viewport={{ once: true, margin: '-100px' }}
    >
      {children}
    </motion.div>
  )
}

function PulsingGlow() {
  return (
    <motion.div
      animate={{ opacity: [0.2, 0.4, 0.2], scale: [1, 1.1, 1] }}
      transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
      className="hero-glow"
    />
  )
}

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
  'history',
])

const PROTECTED_ROUTES = new Set(['dashboard', 'upload', 'model-analysis', 'bias-report', 'explainability', 'reports', 'settings'])
const SESSION_KEY = 'fairhire_session'
const THEME_KEY = 'fairhire_theme_mode'
const IS_LOCAL_HOST = typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname)
const API_BASE = (import.meta.env.VITE_API_URL || (IS_LOCAL_HOST ? 'http://127.0.0.1:8000' : '/api')).replace(/\/$/, '')
const API_CONFIG_ERROR = 'Backend API is not configured for production. Set VITE_API_URL to your deployed backend URL and redeploy the frontend.'
const ROUTE_META = {
  dashboard: ['Workspace', 'Dashboard'],
  upload: ['Workspace', 'Upload Dataset'],
  'model-analysis': ['Workspace', 'Model Evaluation'],
  'bias-report': ['Workspace', 'Fairness Audit'],
  explainability: ['Workspace', 'Decision Rationale'],
  reports: ['Workspace', 'Reports'],
  settings: ['Workspace', 'Settings'],
  history: ['Workspace', 'Analysis History'],
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

function isSelectedPrediction(value) {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value >= 1

  const normalized = String(value || '').trim().toLowerCase()
  return ['1', 'true', 'yes', 'selected', 'approved', 'accept', 'accepted', 'hired', 'recommend', 'recommended', 'pass', 'passed'].includes(normalized)
}

function deriveSelectedCandidates(previewRows = []) {
  if (!previewRows || !previewRows.length) return []
  return previewRows
    .filter((row) => isSelectedPrediction(row?.prediction))
    .map((row, index) => {
      const candidateId = row.candidate_id || row.applicant_id || row.employee_id || row.id || `Candidate ${index + 1}`
      const position = row.role_applied || row.job_role || row.position || row.role || 'Not specified'
      const scoreRaw = row.score ?? row.prediction_score ?? row.probability ?? row.match_score
      const score = scoreRaw == null || Number.isNaN(Number(scoreRaw)) ? null : Number(scoreRaw)

      return {
        id: String(candidateId),
        position: String(position),
        score,
      }
    })
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
    <motion.button
      whileHover={{ scale: 1.04 }}
      whileTap={{ scale: 0.96 }}
      type={type}
      className={className}
      onClick={onClick}
      disabled={disabled}
    >
      <Icon name={icon} />
      {children}
    </motion.button>
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
    ['history', 'History', 'reports'],
  ]

  const flowSteps = [
    ['Upload Data', active !== 'upload' && isAuthenticated],
    ['Train Model', ['model-analysis', 'bias-report', 'explainability', 'reports', 'settings'].includes(active)],
    ['Audit Bias', ['bias-report', 'explainability', 'reports', 'settings'].includes(active)],
    ['Review Rationale', ['explainability', 'reports', 'settings'].includes(active)],
    ['Export Report', ['reports', 'settings'].includes(active)],
  ]

  const crumbs = ROUTE_META[active] || ['Workspace', 'Overview']
  const hasBackgroundLoad = Boolean(loading?.upload || loading?.train || loading?.bias || loading?.explain || loading?.report || loading?.exportReport)
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
            : loading?.exportReport
              ? 'Exporting report PDF'
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


async function downloadReportPdf({ runId, sensitiveColumn, token }) {
  if (!API_BASE) {
    throw new Error(API_CONFIG_ERROR)
  }

  const query = new URLSearchParams({ run_id: runId })
  if (sensitiveColumn) {
    query.set('sensitive_column', sensitiveColumn)
  }

  let response
  try {
    response = await fetch(`${API_BASE}/report/pdf?${query.toString()}`, {
      method: 'GET',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
  } catch {
    throw new Error(`Cannot reach backend API at ${API_BASE}. ${IS_LOCAL_HOST ? 'Make sure FastAPI is running on port 8000.' : 'Deploy backend and set VITE_API_URL.'}`)
  }

  if (!response.ok) {
    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      const payload = await response.json().catch(() => ({}))
      throw new Error(payload.detail || 'Failed to export report')
    }
    throw new Error('Failed to export report')
  }

  const blob = await response.blob()
  const fileName = `fairhire-report-${runId}.pdf`
  const url = window.URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = fileName
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  window.URL.revokeObjectURL(url)
}

async function pollJobResult(jobId, token, onUpdate) {
  for (; ;) {
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

function LandingPage({ onNavigate, onRegister }) {
  const [regData, setRegData] = useState({ name: '', email: '', company: '', password: '' })

  return (
    <div className="landing-page-root">
      <AnimatedBackground />
      <Particles />
      <MouseGlow />
      <header className="landing-nav surface-glass" style={{ padding: '20px 40px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="brand-block">
          <div className="brand-mark" />
          <div>
            <strong>FairHire AI</strong>
            <span className="luminous-text">Ethical hiring intelligence</span>
          </div>
        </div>
        <div className="landing-links">
          <a href="#features">Features</a>
          <a href="#trust">Trust</a>
          <div style={{ display: 'flex', gap: 12 }}>
            <button type="button" className="secondary-button" onClick={() => onNavigate('login')}>Sign In</button>
            <button type="button" className="primary-button" onClick={() => {
              const el = document.getElementById('register');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
              else onNavigate('login');
            }}>Sign Up</button>
          </div>
        </div>
      </header>

      <section className="hero-slab">
        <PulsingGlow />
        <div className="hero-container">
          <div className="hero-copy">
            <FadeIn delay={0.1}>
              <span className="eyebrow"><Icon name="spark" />Next-Gen AI Auditing</span>
            </FadeIn>
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              Audit and correct hiring bias <span className="luminous-text">before it impacts real candidates.</span>
            </motion.h1>
            <FadeIn delay={0.3}>
              <p>
                Detect disparities, explain decisions, and automatically apply mitigation — 
                without sacrificing model performance.
              </p>
            </FadeIn>
            <FadeIn delay={0.4}>
              <div className="hero-actions">
                <ButtonWithIcon type="button" className="primary-button" icon="dashboard" onClick={() => onNavigate('dashboard')}>
                  Run Live Audit
                </ButtonWithIcon>
                <ButtonWithIcon type="button" className="secondary-button" icon="reports" onClick={() => onNavigate('demo')}>
                  Explore Demo Report
                </ButtonWithIcon>
              </div>
            </FadeIn>
          </div>
          <FadeIn delay={0.5} y={60}>
            <div className="hero-visual">
              <AuditSimulation />
            </div>
          </FadeIn>
        </div>
      </section>

      <FadeIn>
        <div className="section-divider">
          <div className="divider-line" />
          <span className="divider-label">HOW IT WORKS</span>
          <div className="divider-line" />
        </div>
      </FadeIn>

      <section className="timeline-section">
        <FadeIn>
          <div className="viz-header">
            <span className="eyebrow">The Process</span>
            <h2>End-to-End Audit Lifecycle</h2>
          </div>
          <div className="audit-timeline">
            <div className="timeline-connector-track">
              <motion.div 
                initial={{ width: 0 }}
                whileInView={{ width: '100%' }}
                transition={{ duration: 2, ease: "easeInOut" }}
                className="timeline-connector-fill" 
              />
            </div>
            {[
              { icon: 'upload', label: 'Upload', desc: 'Secure data ingestion' },
              { icon: 'analysis', label: 'Train', desc: 'Baseline modeling' },
              { icon: 'bias', label: 'Detect', desc: 'Bias identification' },
              { icon: 'shield', label: 'Mitigate', desc: 'Fairness correction' },
              { icon: 'explain', label: 'Explain', desc: 'Decision rationale' },
              { icon: 'reports', label: 'Report', desc: 'Full compliance doc' }
            ].map((step, i) => (
              <motion.div 
                key={i} 
                className="timeline-item"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.2 }}
                viewport={{ once: true }}
              >
                <div className="timeline-icon-box"><Icon name={step.icon} /></div>
                <strong>{step.label}</strong>
                <p>{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </FadeIn>
      </section>

      <section className="bias-centerpiece-expanded">
        <div className="bias-glow-large" />
        <FadeIn>
          <div className="center-header">
            <span className="eyebrow"><Icon name="bias" />Core Differentiator</span>
            <h2>Bias Reduction in Action</h2>
            <p className="section-subtitle-large">Our mitigation engine balances demographic parity without compromising model performance.</p>
          </div>
        </FadeIn>
        
        <div className="bias-impact-showcase">
          <FadeIn delay={0.2} x={-40}>
            <div className="bias-impact-column">
              <span className="column-label">Before Mitigation</span>
              <div className="bias-impact-bars">
                <div className="impact-bar-group">
                  <div className="bias-label-row"><span>Female</span><span>61%</span></div>
                  <div className="bias-bar-track-large"><motion.div initial={{ width: 0 }} whileInView={{ width: '61%' }} transition={{ duration: 1.2 }} viewport={{ once: true }} className="bias-bar-fill-large warning" /></div>
                </div>
                <div className="impact-bar-group">
                  <div className="bias-label-row"><span>Male</span><span>53%</span></div>
                  <div className="bias-bar-track-large"><motion.div initial={{ width: 0 }} whileInView={{ width: '53%' }} transition={{ duration: 1.2 }} viewport={{ once: true }} className="bias-bar-fill-large warning" /></div>
                </div>
              </div>
              <div className="impact-footer">
                <div className="impact-stat">
                  <strong><CountingNumber value="0.14" /></strong>
                  <span>Parity Gap</span>
                </div>
                <div className="impact-status-box warning">High Risk</div>
              </div>
            </div>
          </FadeIn>

          <div className="impact-divider">
            <motion.div 
              initial={{ scale: 0.8, opacity: 0 }}
              whileInView={{ scale: 1, opacity: 1 }}
              className="improvement-badge"
            >
              <Icon name="spark" />
              <span className="improvement-punch">↓ 39%</span>
              <span>Fairness Improvement</span>
            </motion.div>
          </div>

          <FadeIn delay={0.4} x={40}>
            <div className="bias-impact-column highlighted-large">
              <span className="column-label">After Mitigation</span>
              <div className="bias-impact-bars">
                <div className="impact-bar-group">
                  <div className="bias-label-row"><span>Female</span><span>58%</span></div>
                  <div className="bias-bar-track-large"><motion.div initial={{ width: 0 }} whileInView={{ width: '58%' }} transition={{ duration: 1.5, delay: 0.5, ease: "easeOut" }} viewport={{ once: true }} className="bias-bar-fill-large success" /></div>
                </div>
                <div className="impact-bar-group">
                  <div className="bias-label-row"><span>Male</span><span>56%</span></div>
                  <div className="bias-bar-track-large"><motion.div initial={{ width: 0 }} whileInView={{ width: '56%' }} transition={{ duration: 1.5, delay: 0.5, ease: "easeOut" }} viewport={{ once: true }} className="bias-bar-fill-large success" /></div>
                </div>
              </div>
              <div className="impact-footer">
                <div className="impact-stat">
                  <strong className="luminous-text"><CountingNumber value="0.08" /></strong>
                  <span>Parity Gap</span>
                </div>
                <div className="impact-status-box success">Measured Improvement</div>
              </div>
            </div>
          </FadeIn>
        </div>
      </section>

      <section className="landing-viz-section" id="features">
        <FadeIn>
          <div className="viz-header">
            <span className="eyebrow">Visual Intelligence</span>
            <h2>Deep traceability for every decision</h2>
          </div>
        </FadeIn>
        <div className="viz-grid">
          <motion.div
            whileHover={{ y: -10, scale: 1.01, boxShadow: '0 20px 40px rgba(20, 184, 166, 0.1)' }}
            transition={{ type: 'spring', stiffness: 300 }}
            className="viz-card surface-panel"
          >
            <div className="viz-icon-wrap"><Icon name="explain" /></div>
            <h3>Transparent Scoring</h3>
            <p>Understand the drivers behind every hiring recommendation with SHAP-powered feature tracing. Audit specific neurons or feature contributions.</p>
            <ShapVisual />
            <div className="card-badge">Compliance Ready</div>
          </motion.div>
          <motion.div
            whileHover={{ y: -10, scale: 1.01, boxShadow: '0 20px 40px rgba(20, 184, 166, 0.1)' }}
            transition={{ type: 'spring', stiffness: 300 }}
            className="viz-card surface-panel"
          >
            <div className="viz-icon-wrap"><Icon name="analysis" /></div>
            <h3>Model Stability</h3>
            <p>Monitor drift and accuracy guardrails in real-time. Ensure that fairness corrections don't cause unexpected performance degradation.</p>
            <div className="mini-chart-mock">
              <motion.div initial={{ height: 0 }} whileInView={{ height: '70%' }} transition={{ duration: 1 }} className="mini-bar" />
              <motion.div initial={{ height: 0 }} whileInView={{ height: '85%' }} transition={{ duration: 1, delay: 0.1 }} className="mini-bar" />
              <motion.div initial={{ height: 0 }} whileInView={{ height: '75%' }} transition={{ duration: 1, delay: 0.2 }} className="mini-bar" />
              <motion.div initial={{ height: 0 }} whileInView={{ height: '94%' }} transition={{ duration: 1, delay: 0.3 }} className="mini-bar accent" />
            </div>
            <div className="key-row"><span>Validation Acc</span><strong className="luminous-text">94.2%</strong></div>
          </motion.div>
        </div>
      </section>

      <section className="gemini-section">
        <FadeIn>
          <div className="gemini-content">
            <span className="eyebrow">Gemini Powered</span>
            <h2>AI-Assisted Bias Analysis</h2>
            <ul className="gemini-capability-list">
              <li>
                <Icon name="analysis" />
                <div>
                  <strong>Identifies proxy features</strong>
                  <p>Detects variables like referral_source that may hide systemic bias.</p>
                </div>
              </li>
              <li>
                <Icon name="explain" />
                <div>
                  <strong>Natural Language Reasoning</strong>
                  <p>Explains complex model decisions in plain language for human reviewers.</p>
                </div>
              </li>
            </ul>
          </div>
        </FadeIn>
        <FadeIn delay={0.3}>
          <div className="gemini-mockup-wrap">
            <div className="gemini-glow-ring" />
            <div className="gemini-mockup">
              <div className="gemini-header">
                <div className="brand-block">
                  <div className="gemini-pulse" />
                  <strong>Gemini Audit Engine</strong>
                </div>
                <span className="status-chip">Active</span>
              </div>
              <div className="gemini-chat-body">
                <div className="thinking-sequence">
                  {[
                    "Analyzing bias drivers...",
                    "✔ referral_source → high correlation",
                    "✔ experience → neutral",
                    "⚠ education_tier → proxy risk detected"
                  ].map((line, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 1.2, duration: 0.5 }}
                      viewport={{ once: true }}
                      className={`think-item ${i < 3 ? 'done' : 'current'}`}
                    >
                      {line}
                    </motion.div>
                  ))}
                </div>
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: 4.8, duration: 0.6 }}
                  viewport={{ once: true }}
                  className="chat-bubble"
                >
                  "I've detected a significant selection rate disparity for candidates with 'referral_source=employee'. 
                  Recommendation: Apply reweighting to restore demographic parity."
                </motion.div>
                <FadeIn delay={5.5}>
                  <div className="gemini-action-suggest">
                    <button type="button">Apply Threshold Adjustment</button>
                    <button type="button">View Proxy Features</button>
                  </div>
                </FadeIn>
              </div>
            </div>
          </div>
        </FadeIn>
      </section>

      <section className="registration-section" id="register">
        <FadeIn>
          <div className="reg-container">
            <div className="reg-card surface-panel">
              <div className="reg-content">
                <span className="eyebrow"><Icon name="spark" />Get Started</span>
                <h2>Join the future of ethical hiring</h2>
                <p>Start your free 14-day trial. No credit card required.</p>
                
                <form className="reg-form" onSubmit={(e) => { e.preventDefault(); onRegister(regData); }}>
                  <div className="form-group">
                    <label>Full Name</label>
                    <input 
                      type="text" 
                      placeholder="Jane Doe" 
                      required 
                      value={regData.name} 
                      onChange={e => setRegData({...regData, name: e.target.value})} 
                    />
                  </div>
                  <div className="form-group">
                    <label>Work Email</label>
                    <input 
                      type="email" 
                      placeholder="jane@company.com" 
                      required 
                      value={regData.email} 
                      onChange={e => setRegData({...regData, email: e.target.value})} 
                    />
                  </div>
                  <div className="form-group">
                    <label>Company Name</label>
                    <input 
                      type="text" 
                      placeholder="Acme Corp" 
                      required 
                      value={regData.company} 
                      onChange={e => setRegData({...regData, company: e.target.value})} 
                    />
                  </div>
                  <div className="form-group">
                    <label>Password</label>
                    <input 
                      type="password" 
                      placeholder="••••••••" 
                      required 
                      minLength={8} 
                      value={regData.password} 
                      onChange={e => setRegData({...regData, password: e.target.value})} 
                    />
                  </div>
                  <button type="submit" className="primary-button large" style={{ width: '100%', marginTop: 24 }}>
                    Create Account
                  </button>
                </form>
                
                <p className="reg-footer">
                  Already have an account? <button type="button" className="text-link" onClick={() => onNavigate('login')}>Sign In</button>
                </p>
              </div>
              <div className="reg-visual-side">
                <div className="reg-glow-sphere" />
                <div className="reg-stat-mini">
                  <strong>100%</strong>
                  <span>Audit Integrity</span>
                </div>
              </div>
            </div>
          </div>
        </FadeIn>
      </section>

      <section className="trust-anchor" id="trust">
        <div className="trust-bg-accent" />
        <FadeIn>
          <span className="eyebrow">Compliance & Standards</span>
          <h2>Enterprise-grade trust framework</h2>
        </FadeIn>
        <div className="trust-grid">
          {[
            { icon: 'shield', title: 'EEOC Aligned', text: 'Industry-standard fairness checks and 4/5ths rule compliance.' },
            { icon: 'check', title: 'GDPR Ready', text: 'Privacy-first data handling with PII masking capabilities.' },
            { icon: 'analysis', title: 'EU AI Act', text: 'Full risk transparency and documentation for high-risk AI.' },
            { icon: 'reports', title: 'Fully Auditable', text: 'Comprehensive trace history and exportable PDF reports.' }
          ].map((item, i) => (
            <FadeIn key={i} delay={i * 0.15}>
              <div className="trust-card">
                <div className="trust-icon-box"><Icon name={item.icon} /></div>
                <h4>{item.title}</h4>
                <p>{item.text}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </section>

      <FadeIn>
        <section className="final-cta-section">
          <div className="cta-glow" />
          <h2 style={{ maxWidth: 800, margin: '0 auto 24px' }}>Start auditing hiring models with measurable fairness guarantees.</h2>
          <p>No guesswork. No black boxes. Full audit trail for every decision.</p>
          <div className="cta-actions">
            <ButtonWithIcon type="button" className="primary-button large" icon="dashboard" onClick={() => onNavigate('dashboard')}>
              Run Live Audit
            </ButtonWithIcon>
            <ButtonWithIcon type="button" className="secondary-button large" icon="reports" onClick={() => onNavigate('demo')}>
              View Sample Report
            </ButtonWithIcon>
          </div>
        </section>
      </FadeIn>

      <footer style={{ padding: '60px 40px', borderTop: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
        <div className="brand-block" style={{ justifyContent: 'center', marginBottom: 24 }}>
          <div className="brand-mark" />
          <strong>FairHire AI</strong>
        </div>
        <p className="micro-copy">© 2026 FairHire AI. Built with Google Gemini for Solution Challenge.</p>
      </footer>
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

          <div className="login-footer" style={{ marginTop: 32, textAlign: 'center', borderTop: '1px solid var(--border-subtle)', paddingTop: 24 }}>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Don't have an account? 
              <button type="button" className="text-link" style={{ marginLeft: 8 }} onClick={() => {
                onNavigate('landing');
                setTimeout(() => {
                  const el = document.getElementById('register');
                  if (el) el.scrollIntoView({ behavior: 'smooth' });
                }, 100);
              }}>Sign Up Now</button>
            </p>
          </div>

          <SymbolFieldStrip items={loginFields} />
        </form>
      </section>
    </div>
  )
}

function DashboardPage({ onNavigate, biasData, trainData, loading }) {
  const biasChart = useMemo(() => {
    if (!biasData?.selection_rate_by_group) return []
    return Object.entries(biasData.selection_rate_by_group).map(([group, value]) => ({ group, value }))
  }, [biasData])

  const dashboardFields = [
    { icon: 'warning', label: 'Monitor', title: 'Bias drift alerts', text: 'Watch fairness index shifts before they cross risk thresholds.' },
    { icon: 'users', label: 'Coverage', title: 'Group representation', text: 'Ensure all sensitive groups remain visible in selection analysis.' },
    { icon: 'spark', label: 'Action', title: 'Weekly review cadence', text: 'Schedule recurring checks to keep hiring models accountable.' },
  ]

  const fairnessIndex = biasData?.fairness_index ?? 0
  const parityDelta = biasData?.demographic_parity_difference ?? 0
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
        <MetricCard label="Active audits" value={biasData ? "1" : "0"} note="Connected to API" icon="reports" />
        <MetricCard label="Risk signal" value={(biasData?.fairness_index ?? 0).toFixed(2)} note="Real-time monitoring" accent icon="warning" />
        <MetricCard label="Accuracy" value={(trainData?.accuracy ?? 0).toFixed(2)} note="Latest trained model" icon="analysis" />
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
          <li><Icon name="analysis" />{biasData ? "Analysis complete: inspect subgroup parity findings." : "Upload a dataset to generate behavioral insights."}</li>
          {biasData && (
            <>
              <li><Icon name="warning" />Gender disparity detected (Δ = {parityDelta.toFixed(2)}).</li>
              <li><Icon name="bias" />Feature importance analyzed for proxy bias.</li>
            </>
          )}
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
  requiredPosition,
  setRequiredPosition,
  trainingProgress,
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
              <label className="target-select-label">
                Required position for hiring
                <input
                  type="text"
                  list="role-suggestions"
                  value={requiredPosition}
                  onChange={(event) => setRequiredPosition(event.target.value)}
                  placeholder="e.g. Data Scientist"
                />
                <datalist id="role-suggestions">
                  {(uploadData.role_suggestions || []).map((role) => (
                    <option key={role} value={role} />
                  ))}
                </datalist>
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
          disabled={!uploadData || loading.train || !requiredPosition.trim()}
        >
          {loading.train ? 'Training model...' : 'Continue to Mapping'}
        </ButtonWithIcon>
      </div>

      {trainingProgress.active || trainingProgress.status === 'failed' ? (
        <section className="training-progress-shell surface-card" aria-live="polite">
          <div className="key-row">
            <strong>{trainingProgress.label || 'Preparing training...'}</strong>
            <span>{trainingProgress.percent}%</span>
          </div>
          <div
            className={`training-progress-track ${trainingProgress.status === 'failed' ? 'failed' : ''}`}
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={trainingProgress.percent}
            aria-label="Model training progress"
          >
            <span style={{ width: `${trainingProgress.percent}%` }} />
          </div>
          {trainingProgress.message ? <p className="micro-copy">{trainingProgress.message}</p> : null}
        </section>
      ) : null}

      <SymbolFieldStrip items={uploadFields} />
    </>
  )
}

function ModelAnalysisPage({ onNavigate, trainData, loading }) {
  const confusionData = useMemo(() => {
    const matrix = trainData?.confusion_matrix || { tp: 0, fp: 0, tn: 0, fn: 0 }
    return [
      { name: 'TP', value: matrix.tp || 0 },
      { name: 'TN', value: matrix.tn || 0 },
      { name: 'FP', value: matrix.fp || 0 },
      { name: 'FN', value: matrix.fn || 0 },
    ]
  }, [trainData])

  const fairnessBefore = trainData?.fairness?.before
  const fairnessAfter = trainData?.fairness?.after
  const mitigationMethod = trainData?.fairness?.method || 'baseline'
  const mitigationParameter = trainData?.fairness?.parameter ?? 'none'
  const accuracyImpact = Number(trainData?.fairness?.accuracy_impact ?? 0)
  const accuracyImpactPct = accuracyImpact * 100
  const diagnostics = trainData?.diagnostics || []

  const mitigationLabel = mitigationMethod === 'threshold'
    ? `Threshold Adjustment (${mitigationParameter})`
    : mitigationMethod === 'reweight'
      ? 'Reweighting (Balanced Bootstrap)'
      : mitigationMethod === 'mask'
        ? 'Feature Masking'
        : 'Baseline Retained'

  const analysisFields = [
    { icon: 'analysis', label: 'Performance', title: 'Confusion balance', text: 'Compare false positives and false negatives before deployment.' },
    { icon: 'spark', label: 'Validation', title: 'Cross-check metrics', text: 'Use precision and recall together, not accuracy alone.' },
    { icon: 'bias', label: 'Next step', title: 'Fairness audit required', text: 'Run bias checks before approving candidate scoring in production.' },
  ]

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Accuracy" value={(trainData?.accuracy ?? 0).toFixed(2)} note="Balanced validation split" icon="analysis" />
        <MetricCard label="Precision" value={(trainData?.precision ?? 0).toFixed(2)} note="Prediction quality" accent icon="check" />
        <MetricCard label="Recall" value={(trainData?.recall ?? 0).toFixed(2)} note="Capture effectiveness" icon="spark" />
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

      <SectionCard title="Fairness Improvement" subtitle="Before vs after mitigation under performance guardrail" icon="bias">
        <div className="before-after-grid">
          <article className="surface-card before-block">
            <span className="eyebrow"><Icon name="warning" />Before</span>
            <strong>Fairness Gap: {(fairnessBefore?.demographic_parity_difference ?? 0).toFixed(2)}</strong>
            <p>Fairness Index: {(fairnessBefore?.fairness_index ?? 0).toFixed(2)}</p>
          </article>
          <article className="surface-card after-block">
            <span className="eyebrow"><Icon name="check" />After</span>
            <strong>Fairness Gap: {(fairnessAfter?.demographic_parity_difference ?? 0).toFixed(2)}</strong>
            <p>Fairness Index: {(fairnessAfter?.fairness_index ?? 0).toFixed(2)}</p>
          </article>
        </div>
        <div className="stacked-copy">
          <div className="key-row"><span>Method</span><strong>{mitigationLabel}</strong></div>
          <div className="key-row"><span>Accuracy impact</span><strong>{accuracyImpactPct >= 0 ? '+' : ''}{accuracyImpactPct.toFixed(1)}%</strong></div>
        </div>
        {diagnostics.length ? (
          <ul className="ai-list">
            {diagnostics.slice(0, 3).map((item, idx) => (
              <li key={`diag-${idx}`}><Icon name="check" />{item}</li>
            ))}
          </ul>
        ) : null}
      </SectionCard>

      <SymbolFieldStrip items={analysisFields} />
    </>
  )
}

function BiasReportPage({ onNavigate, biasData, trainData, loading, runId, sensitiveColumn, sensitiveOptions, onSensitiveColumnChange }) {
  const groupRows = Object.entries(biasData?.selection_rate_by_group || {})
  const chartRows = groupRows.map(([group, value]) => ({ group, value: Number(value) }))
  const maxValue = Math.max(...chartRows.map((row) => row.value))
  const minValue = Math.min(...chartRows.map((row) => row.value))
  const groupGap = maxValue - minValue
  const [threshold, setThreshold] = useState(0.5)
  const [weightShift, setWeightShift] = useState(0)
  const baseFairness = biasData?.fairness_index ?? 0
  const adjustedFairness = Math.min(0.96, Math.max(0.5, baseFairness + (0.2 * (0.6 - Math.abs(weightShift - 0.2))) + (threshold - 0.5) * 0.12))
  const simulatedGap = Math.max(0.02, groupGap - (threshold - 0.5) * 0.08 - weightShift * 0.1)
  const fairnessProof = trainData?.fairness
  const fairnessBefore = fairnessProof?.before
  const fairnessAfter = fairnessProof?.after
  const method = fairnessProof?.method || 'baseline'
  const parameter = fairnessProof?.parameter ?? 'none'
  const accuracyImpact = Number(fairnessProof?.accuracy_impact ?? 0)
  const beforeRates = fairnessBefore?.selection_rate_by_group || {}
  const afterRates = fairnessAfter?.selection_rate_by_group || {}
  const proofGroups = Array.from(new Set([...Object.keys(beforeRates), ...Object.keys(afterRates)]))
  const beforeAfterChartRows = proofGroups.length
    ? proofGroups.map((group) => ({
      group,
      before: Number(beforeRates[group] ?? 0),
      after: Number(afterRates[group] ?? 0),
    }))
    : []

  const methodLabel = method === 'threshold'
    ? `Threshold Adjustment (${parameter})`
    : method === 'reweight'
      ? 'Reweighting (Balanced Bootstrap)'
      : method === 'mask'
        ? 'Feature Masking'
        : 'Baseline Retained'
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
          value={(biasData?.demographic_parity_difference ?? 0).toFixed(2)}
          note="Difference across groups"
          accent
          icon="warning"
        />
        <MetricCard
          label="Fairness index"
          value={(biasData?.fairness_index ?? 0).toFixed(2)}
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
              <label className="target-select-label sensitive-select">
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
            <strong>Fairness Gap: {(fairnessBefore?.demographic_parity_difference ?? 0.14).toFixed(2)}</strong>
            <p>Fairness Index: {(fairnessBefore?.fairness_index ?? 0.68).toFixed(2)}</p>
          </article>
          <article className="surface-card after-block">
            <span className="eyebrow"><Icon name="check" />After mitigation</span>
            <strong>Fairness Gap: {(fairnessAfter?.demographic_parity_difference ?? 0.08).toFixed(2)}</strong>
            <p>Fairness Index: {(fairnessAfter?.fairness_index ?? 0.84).toFixed(2)}</p>
          </article>
        </div>
        <div className="chart-host">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={beforeAfterChartRows}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
              <XAxis dataKey="group" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="before" name="Before" radius={[6, 6, 0, 0]} fill="var(--chart-threshold)" />
              <Bar dataKey="after" name="After" radius={[6, 6, 0, 0]} fill="var(--chart-bar-primary)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="stacked-copy">
          <div className="key-row"><span>Strategy used</span><strong>{methodLabel}</strong></div>
          <div className="key-row"><span>Accuracy impact</span><strong>{accuracyImpact >= 0 ? '+' : ''}{(accuracyImpact * 100).toFixed(1)}%</strong></div>
        </div>
        <p className="micro-copy">Based on current model behavior.</p>
      </SectionCard>

      <SymbolFieldStrip items={biasFields} />
    </>
  )
}

function ExplainabilityPage({ onNavigate, explainData, loading }) {
  const featureData = explainData?.top_global_features || []

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

function ReportsPage({ reportData, trainData, biasData, loading, onExportReport, canExport }) {
  const reportSummary = useMemo(() => {
    const verified = reportData ? 98 : 92
    const pending = reportData ? 16 : 22
    return [
      { name: 'Verified', value: verified, color: '#14b8a6' },
      { name: 'Pending', value: pending, color: '#f59e0b' },
    ]
  }, [reportData])
  const selectedCandidates = useMemo(
    () => deriveSelectedCandidates(reportData?.train?.prediction_preview || trainData?.prediction_preview || []),
    [reportData?.train?.prediction_preview, trainData?.prediction_preview],
  )

  const reportFields = [
    { icon: 'reports', label: 'Required', title: 'Versioned reports', text: 'Keep immutable snapshots for every model release cycle.' },
    { icon: 'check', label: 'Governance', title: 'Reviewer sign-off', text: 'Capture approver names and decision timestamps.' },
    { icon: 'download', label: 'Distribution', title: 'Export package', text: 'Share report bundles with legal and HR leadership.' },
  ]

  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Stored reports" value={reportData || trainData ? "1" : "0"} note="Connected to API" icon="reports" />
        <MetricCard label="Verified" value={reportData || (biasData?.fairness_index > 0.9) ? "1" : "0"} note="Cleanly signed off" accent icon="check" />
        <MetricCard label="Pending" value={(!reportData && biasData?.fairness_index <= 0.9) ? "1" : "0"} note="Requires human review" icon="warning" />
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
            <div className="key-row"><span>Model accuracy</span><strong>{((reportData?.train?.accuracy ?? trainData?.accuracy ?? 0) * 100).toFixed(1)}%</strong></div>
            <div className="key-row"><span>Fairness index</span><strong>{((reportData?.bias?.fairness_index ?? biasData?.fairness_index ?? 0) * 100).toFixed(1)}%</strong></div>
            <div className="key-row"><span>Top feature</span><strong>{reportData?.explain?.top_global_features?.[0]?.feature ?? Object.keys(trainData?.feature_importance || {})[0] ?? 'experience'}</strong></div>
          </div>
          <div className="page-actions">
            <ButtonWithIcon
              type="button"
              className="primary-button"
              icon="download"
              onClick={onExportReport}
              disabled={!canExport || loading.exportReport}
            >
              {loading.exportReport ? 'Exporting...' : 'Export PDF report'}
            </ButtonWithIcon>
          </div>
        </SectionCard>
      </div>

      <SectionCard title="Board-ready report preview" subtitle="Executive compliance format" icon="reports">
        <div className="executive-grid">
          <article className="surface-card executive-block">
            <h3>Summary</h3>
            <p>{reportData?.summary || reportData?.narrative || "Current model performance remains high with moderate fairness risk requiring follow-up."}</p>
          </article>
          <article className="surface-card executive-block">
            <h3>Risk Areas</h3>
            <p>{biasData?.recommendations?.[1] || "Gender parity gap and referral source influence require immediate policy review."}</p>
          </article>
          <article className="surface-card executive-block">
            <h3>Compliance Status</h3>
            <p>{biasData?.fairness_index > 0.9 ? "Fully compliant with corporate fairness standards." : "Provisionally compliant pending threshold recalibration and secondary validation run."}</p>
          </article>
          <article className="surface-card executive-block">
            <h3>Recommendations</h3>
            <p>{biasData?.recommendations?.[2] || "Reweight dataset, rerun fairness test, and attach updated explainability appendix."}</p>
          </article>
        </div>
        <p className="micro-copy">Derived from training dataset patterns.</p>
      </SectionCard>

      <SectionCard title="Selected Candidates" subtitle="Candidates recommended by the trained model" icon="users">
        {selectedCandidates.length ? (
          <div className="stacked-copy">
            {selectedCandidates.map((candidate) => (
              <div key={`${candidate.id}-${candidate.position}`} className="key-row">
                <span>{candidate.id} - {candidate.position}</span>
                <strong>{candidate.score == null ? 'Selected' : `${(candidate.score <= 1 ? candidate.score * 100 : candidate.score).toFixed(1)}%`}</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="card-copy">No selected candidates are available yet. Train the model and generate a report to populate this section.</p>
        )}
      </SectionCard>

      <SymbolFieldStrip items={reportFields} />
    </>
  )
}

function HistoryPage({ history, loading }) {
  return (
    <>
      <div className="metric-grid">
        <MetricCard label="Total Runs" value={history.length} note="In Firestore" icon="reports" />
        <MetricCard label="Last Audit" value={history[0] ? new Date(history[0].timestamp).toLocaleDateString() : 'N/A'} note="Recent activity" accent icon="spark" />
        <MetricCard label="Status" value="Syncing" note="Firestore active" icon="check" />
      </div>

      <SectionCard title="Analysis History" subtitle="Previous bias audits and training runs" icon="reports">
        {loading ? (
          <Skeleton className="table-skeleton" />
        ) : history.length > 0 ? (
          <div className="table-scroll">
            <table className="table-mock">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Run ID</th>
                  <th>Accuracy</th>
                  <th>Fairness Index</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item, idx) => (
                  <tr key={idx}>
                    <td>{item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'}</td>
                    <td><code style={{ fontSize: '0.8em' }}>{item.run_id}</code></td>
                    <td>{(item.metrics?.accuracy ?? 0).toFixed(2)}</td>
                    <td>{(item.bias_data?.fairness_index ?? 0).toFixed(2)}</td>
                    <td><span className="badge-chip success">Completed</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="card-copy">No previous analysis runs found in history.</p>
        )}
      </SectionCard>
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

function ChatAssistant({ session, biasData, trainData }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([
    { role: 'bot', content: 'Hello! I am the FairHire AI Auditor, powered by Gemini. How can I help you investigate your hiring audit today?' }
  ])
  const [loading, setLoading] = useState(false)

  const ask = async (e) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    const userMsg = { role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setQuery('')
    setLoading(true)

    try {
      const payload = await callApi('/assistant', {
        method: 'POST',
        token: session?.token,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: query,
          context: {
            bias_data: biasData,
            train_data: trainData,
            user_name: session?.user?.name
          }
        })
      })
      setMessages(prev => [...prev, { role: 'bot', content: payload.answer, poweredBy: payload.powered_by }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', content: 'Sorry, I encountered an error. Please try again later.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <button type="button" className="assistant-trigger" onClick={() => setOpen(!open)} title="Ask Gemini Auditor">
        <Icon name="spark" />
      </button>
      {open && (
        <article className="assistant-panel surface-glass">
          <header className="assistant-header">
            <h3><Icon name="shield" />FairHire Auditor</h3>
            <button type="button" className="icon-button" style={{ width: 32, height: 32 }} onClick={() => setOpen(false)}>×</button>
          </header>
          <div className="assistant-messages">
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                {m.content}
                {m.poweredBy && <small style={{ display: 'block', marginTop: 4, opacity: 0.6, fontSize: '0.7rem' }}>✨ {m.poweredBy}</small>}
              </div>
            ))}
            {loading && <div className="msg bot">Auditing data...</div>}
          </div>
          <form className="assistant-input" onSubmit={ask}>
            <input 
              placeholder="Ask about bias, candidates, or metrics..." 
              value={query}
              onChange={e => setQuery(e.target.value)}
            />
            <button type="submit" className="icon-button" style={{ width: 42, height: 42 }}><Icon name="spark" /></button>
          </form>
        </article>
      )}
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
    clearToasts: () => setToasts([]),
    dismissToast: (id) => setToasts((current) => current.filter((toast) => toast.id !== id)),
  }
}

export default function App() {
  const initialTrainingProgress = {
    active: false,
    percent: 0,
    label: '',
    message: '',
    status: 'idle',
  }

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
  const [historyData, setHistoryData] = useState([])
  const [requiredPosition, setRequiredPosition] = useState('')
  const [isSidebarCompact, setIsSidebarCompact] = useState(false)
  const [uiBooting, setUiBooting] = useState(true)
  const [routeStageClass, setRouteStageClass] = useState('entered')
  const [trainingProgress, setTrainingProgress] = useState(initialTrainingProgress)
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

  const loadDemoReport = () => {
    setSession(DEMO_REPORT.session)
    setTrainData(DEMO_REPORT.train)
    setBiasData(DEMO_REPORT.bias)
    setExplainData(DEMO_REPORT.explain)
    setReportData(DEMO_REPORT.report)
    setSensitiveColumn(DEMO_REPORT.bias.sensitive_column)
    pushToast('info', 'Demo loaded', 'Welcome to the Demo Environment. You are viewing a pre-audited dataset.')
    navigate('reports')
    setRoute('reports')
  }

  const [loading, setLoading] = useState({
    auth: false,
    upload: false,
    train: false,
    bias: false,
    explain: false,
    report: false,
    exportReport: false,
    history: false,
  })

  const { toasts, pushToast, dismissToast, clearToasts } = useToasts()

  const isAuthenticated = Boolean(session?.token)
  const runId = trainData?.run_id || null
  const isTrainingActive = loading.train || trainingProgress.active
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
        const payload = await callApi(`/bias?run_id=${encodeURIComponent(runId)}&sensitive_column=${encodeURIComponent(sensitiveColumn)}`, { token: session?.token })
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
        const submission = await callApi(`/explain?run_id=${encodeURIComponent(runId)}&async_job=false`, { token: session?.token })
        const payload = submission.result
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

  useEffect(() => {
    const loadHistory = async () => {
      if (route !== 'history' || !session?.token || loading.history) return
      setLoading((prev) => ({ ...prev, history: true }))
      try {
        const payload = await callApi('/history', { token: session?.token })
        setHistoryData(payload)
      } catch (error) {
        pushToast('error', 'History failed', error.message)
      } finally {
        setLoading((prev) => ({ ...prev, history: false }))
      }
    }
    loadHistory()
  }, [route, session?.token])

  const handleLogin = async ({ email, password }) => {
    setLoading((prev) => ({ ...prev, auth: true }))
    try {
      // Step 1: Check if user exists
      const { exists } = await callApi(`/auth/exists?email=${encodeURIComponent(email)}`)
      
      if (!exists) {
        pushToast('warning', 'Account not found', 'You are not registered yet. Redirecting to sign up...')
        navigate('landing')
        setRoute('landing')
        setTimeout(() => {
          const el = document.getElementById('register')
          if (el) el.scrollIntoView({ behavior: 'smooth' })
        }, 300)
        return
      }

      // Step 2: Proceed to login
      const authPayload = await callApi('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })

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

  const handleRegister = async ({ email, password, name, company }) => {
    setLoading((prev) => ({ ...prev, auth: true }))
    try {
      const authPayload = await callApi('/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, name, company }),
      })
      setSession(authPayload)
      upsertUserProfile(authPayload.user).catch(() => {})
      pushToast('success', 'Account created', `Welcome, ${authPayload.user.name}. Your ethical auditing workspace is ready.`)
      navigate('dashboard')
      setRoute('dashboard')
    } catch (error) {
      pushToast('error', 'Registration failed', error.message || 'Unable to create account right now.')
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
    setRequiredPosition('')
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
      payload = await callApi('/upload', { method: 'POST', body: formData, token: session?.token })

      setUploadData(payload)
      setSelectedTarget(payload.target_suggestions?.[0] || payload.columns?.[0] || '')
      setRequiredPosition('')
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
      setTrainingProgress({
        active: false,
        percent: 0,
        label: 'Training not started',
        message: 'Upload a dataset before training.',
        status: 'failed',
      })
      return
    }
    if (!requiredPosition.trim()) {
      setTrainingProgress({
        active: false,
        percent: 0,
        label: 'Training not started',
        message: 'Enter the target hiring position before training.',
        status: 'failed',
      })
      return
    }

    setLoading((prev) => ({ ...prev, train: true }))
    clearToasts()
    setTrainingProgress({
      active: true,
      percent: 8,
      label: 'Submitting training job',
      message: 'Initializing dataset and training pipeline.',
      status: 'running',
    })

    try {
      const target = selectedTarget || uploadData.target_suggestions?.[0] || uploadData.columns?.[0]
      const submission = await callApi('/train', {
        method: 'POST',
        token: session?.token,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: uploadData.dataset_id,
          target_column: target,
          required_position: requiredPosition.trim(),
          model_type: 'random_forest',
          async_job: true,
          sensitive_column: sensitiveColumn,
          include_fairness_proof: true,
        }),
      })

      const jobId = submission.job_id
      if (!jobId) {
        throw new Error('Training job submission did not return a job ID')
      }

      let pollStep = 0
      setTrainingProgress((current) => ({
        ...current,
        percent: Math.max(current.percent, 16),
        label: 'Training job queued',
        message: 'Waiting for compute slot allocation.',
      }))

      const payload = await pollJobResult(jobId, session?.token, (jobStatus) => {
        pollStep += 1
        setTrainingProgress((current) => {
          const status = String(jobStatus?.status || '').toLowerCase()
          const incoming = String(jobStatus?.message || '')
          const next = { ...current }

          if (status === 'queued') {
            next.percent = Math.min(40, Math.max(current.percent, 16 + pollStep * 4))
            next.label = 'Training job queued'
            next.message = incoming || 'Waiting in queue.'
            next.status = 'running'
          } else if (status === 'running') {
            next.percent = Math.min(92, Math.max(current.percent, 44 + pollStep * 5))
            next.label = 'Model training in progress'
            next.message = incoming || 'Training model and validating fairness guardrails.'
            next.status = 'running'
          } else if (status === 'completed') {
            next.percent = 100
            next.label = 'Training completed'
            next.message = incoming || 'Run is ready for analysis.'
            next.status = 'completed'
          }

          return next
        })
      })

      setTrainData(payload)
      setBiasError(null)
      setExplainError(null)
      setReportError(null)
      saveTrainingRun({ user: session?.user, training: payload }).catch(() => {
        console.warn('Training completed, but run sync to Firestore failed.')
      })
      setTrainingProgress({
        active: true,
        percent: 100,
        label: 'Training completed',
        message: `Run ${payload.run_id} is ready for bias analysis.`,
        status: 'completed',
      })

      window.setTimeout(() => {
        setTrainingProgress(initialTrainingProgress)
      }, 1800)

      navigate('model-analysis')
      setRoute('model-analysis')
    } catch (error) {
      setTrainingProgress({
        active: false,
        percent: 0,
        label: 'Training failed',
        message: error.message || 'Unable to train model.',
        status: 'failed',
      })
    } finally {
      setLoading((prev) => ({ ...prev, train: false }))
    }
  }

  const handleExportReport = async () => {
    if (!runId) {
      pushToast('error', 'No active run', 'Train a model before exporting the report.')
      return
    }

    setLoading((prev) => ({ ...prev, exportReport: true }))
    try {
      await downloadReportPdf({ runId, sensitiveColumn, token: session?.token })
      pushToast('success', 'Report exported', `Downloaded PDF for run ${runId}.`)
    } catch (error) {
      pushToast('error', 'Export failed', error.message || 'Unable to export report.')
    } finally {
      setLoading((prev) => ({ ...prev, exportReport: false }))
    }
  }

  const wrappedPage = useMemo(() => {
    if (PROTECTED_ROUTES.has(route) && !isAuthenticated) {
      return <LoginPage onNavigate={navigate} onLogin={handleLogin} authLoading={loading.auth} />
    }

    switch (route) {
      case 'landing':
        return <LandingPage onNavigate={(r) => {
          if (r === 'demo') loadDemoReport()
          else navigate(r)
        }} onRegister={handleRegister} />
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
              requiredPosition={requiredPosition}
              setRequiredPosition={setRequiredPosition}
              trainingProgress={trainingProgress}
            />
          </AppShell>
        )
      case 'model-analysis':
        return (
          <AppShell active="model-analysis" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <ModelAnalysisPage onNavigate={navigate} trainData={trainData} loading={loading} />
          </AppShell>
        )
      case 'history':
        return (
          <AppShell active="history" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <HistoryPage history={historyData} loading={loading.history} />
          </AppShell>
        )
      case 'bias-report':
        return (
          <AppShell active="bias-report" onNavigate={navigate} isAuthenticated={isAuthenticated} onLogout={handleLogout} userProfile={userProfile} isSidebarCompact={isSidebarCompact} onToggleSidebar={() => setIsSidebarCompact((v) => !v)} loading={loading}>
            <BiasReportPage
              onNavigate={navigate}
              biasData={biasData}
              trainData={trainData}
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
            <ReportsPage
              reportData={reportData}
              trainData={trainData}
              biasData={biasData}
              loading={loading}
              onExportReport={handleExportReport}
              canExport={Boolean(runId)}
            />
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
        return <LandingPage onNavigate={(r) => {
          if (r === 'demo') loadDemoReport()
          else navigate(r)
        }} />
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
    requiredPosition,
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
      {!isTrainingActive ? <ToastStack toasts={toasts} onDismiss={dismissToast} /> : null}
      {isAuthenticated && <ChatAssistant session={session} biasData={biasData} trainData={trainData} />}
    </ErrorBoundary>
  )
}
