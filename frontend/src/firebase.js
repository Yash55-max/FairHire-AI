import { initializeApp } from 'firebase/app'
import { getAnalytics, isSupported } from 'firebase/analytics'
import { getFirestore } from 'firebase/firestore'

// FairHire AI Firebase configuration
const firebaseConfig = {
  apiKey: '',
  authDomain: 'fairhire-67f38.firebaseapp.com',
  projectId: 'fairhire-67f38',
  storageBucket: 'fairhire-67f38.firebasestorage.app',
  messagingSenderId: '390341982598',
  appId: '1:390341982598:web:6d6859ea62a39f15d71407',
  measurementId: 'G-XYSK5DCQ0J',
}

const app = initializeApp(firebaseConfig)
const db = getFirestore(app)

// Analytics requires a browser environment and may be unavailable in some contexts.
const analyticsPromise =
  typeof window === 'undefined'
    ? Promise.resolve(null)
    : isSupported()
        .then((supported) => (supported ? getAnalytics(app) : null))
        .catch(() => null)

export { app, db, analyticsPromise }
