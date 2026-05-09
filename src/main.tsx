import React from 'react'
import { createRoot } from 'react-dom/client'
import VisionAssist from './vision-assist-app'
import './index.css'

function Main() {
  return <VisionAssist />
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Main />
  </React.StrictMode>
)
