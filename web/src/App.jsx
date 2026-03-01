import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import LiveViewDashboard from './live/LiveViewDashboard'
import LiveViewInlineSection from './live/LiveViewInlineSection'
import ScheduleLogTab from './components/tabs/ScheduleLogTab'

const OPERATION_SEQUENCE = ['smt', 'reflow', 'tht', 'aoi', 'test', 'coating', 'pack']
const PHASE_TO_OPERATION = {
  SMT: 'smt',
  REFLOW: 'reflow',
  Reflow: 'reflow',
  THT: 'tht',
  AOI: 'aoi',
  TEST: 'test',
  Test: 'test',
  COATING: 'coating',
  Coating: 'coating',
  PACK: 'pack',
  Pack: 'pack',
}

const MACHINE_LABELS = {
  smt: 'SMT',
  reflow: 'Reflow',
  tht: 'THT',
  aoi: 'AOI',
  test: 'Test',
  coating: 'Coating',
  pack: 'Pack',
}

const PHASE_COLORS = {
  smt:     { bg: '#f7f8fa', border: '#aebdcf', text: '#647b97' },
  reflow:  { bg: '#f7f8fa', border: '#aebdcf', text: '#647b97' },
  tht:     { bg: '#f7f8fa', border: '#aebdcf', text: '#647b97' },
  aoi:     { bg: '#f7f8fa', border: '#aebdcf', text: '#647b97' },
  test:    { bg: '#f7f8fa', border: '#aebdcf', text: '#647b97' },
  coating: { bg: '#f7f8fa', border: '#aebdcf', text: '#647b97' },
  pack:    { bg: '#f7f8fa', border: '#aebdcf', text: '#647b97' },
}

function App() {
  const [schedule, setSchedule] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('gantt')
  const [productionStates, setProductionStates] = useState([])
  const [loadingStates, setLoadingStates] = useState(false)
  const [lineVisionPhase, setLineVisionPhase] = useState('NOT_PRESENT')
  const [productionLog, setProductionLog] = useState([])
  const [pushStatus, setPushStatus] = useState({})
  const [pushing, setPushing] = useState(false)
  const timelineRef = useRef(null)
  const timelineInstance = useRef(null)
  const autoAdvanceRef = useRef({
    inFlight: false,
    lastAttemptKey: '',
    lastAttemptAt: 0,
    lastSuccessKey: '',
  })

  useEffect(() => {
    fetchSchedule()
    fetchProductionStates()
    fetchProductionLog()
    fetchPushStatus()
  }, [])

  // Fetch push status for all orders
  const fetchPushStatus = async () => {
    try {
      const response = await axios.get('/api/production/push-status')
      const map = {}
      for (const r of response.data) {
        map[r.order_number] = r
      }
      setPushStatus(map)
    } catch (err) {
      console.error('Failed to fetch push status', err)
    }
  }

  // Push all orders to Arke
  const pushToArke = async () => {
    setPushing(true)
    try {
      const response = await axios.post('/api/production/push')
      // Refresh push status + log
      await fetchPushStatus()
      await fetchProductionLog()
      const { created, skipped, failed } = response.data
      alert(`Push complete: ${created} created, ${skipped} already pushed, ${failed} failed`)
    } catch (err) {
      console.error('Push to Arke failed', err)
      alert('Push to Arke failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setPushing(false)
    }
  }

  useEffect(() => {
    if (schedule && activeTab === 'gantt' && timelineRef.current && window.vis) {
      initTimeline()
    }
  }, [schedule, activeTab])

  const initTimeline = () => {
    if (!schedule || !timelineRef.current) return

    const items = []
    const groups = [{ id: 'production-line', content: 'Production Line' }]

    const conflictOrders = new Set()
    if (schedule.conflicts && schedule.conflicts.length > 0) {
      schedule.conflicts.forEach(conflict => {
        conflictOrders.add(conflict.priority_first.order)
        conflictOrders.add(conflict.edf_first.order)
      })
    }

    schedule.production_plans.forEach((plan, index) => {
      const startDate = new Date(plan.starts_at)
      const endDate = new Date(plan.ends_at)
      const deadline = new Date(plan.deadline)
      
      const deltaDays = Math.ceil((deadline - endDate) / (1000 * 60 * 60 * 24))
      const isLate = deltaDays < 0
      const deadlineText = `Due: ${deadline.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`
      
      const isConflict = conflictOrders.has(plan.order_number)

      items.push({
        id: `order-${index}`,
        content: `<div class="order-label">
          <strong>${plan.order_number}</strong>
          ${isConflict ? '<span class="conflict-badge">⚠</span>' : ''}
          <br/><span class="order-detail">${plan.quantity}x ${plan.product}</span>
          <br/><span class="delta-due ${isLate ? 'late' : 'on-time'}">${deadlineText}</span>
        </div>`,
        start: startDate,
        end: endDate,
        group: 'production-line',
        className: `order-item priority-${plan.priority} ${isLate ? 'late-order' : ''}`,
        title: `${plan.customer}\n${plan.product}\nQty: ${plan.quantity}\nPriority: P${plan.priority}\n${plan.reasoning}`
      })
    })

    // ── Machine sub-lanes: add phase-level items ──
    const usedOps = new Set()
    schedule.production_plans.forEach((plan, index) => {
      if (!plan.phases || plan.phases.length === 0) return
      plan.phases.forEach((phase, phIdx) => {
        const op = PHASE_TO_OPERATION[phase.name]
        if (!op) return
        usedOps.add(op)
        const colors = PHASE_COLORS[op] || { bg: '#f3f4f6', border: '#9ca3af', text: '#374151' }
        items.push({
          id: `phase-${index}-${phIdx}`,
          content: `<div class="phase-label"><strong>${phase.name}</strong><br/><span class="phase-detail">${plan.order_number} · ${phase.duration_minutes}m</span></div>`,
          start: new Date(phase.starts_at),
          end: new Date(phase.ends_at),
          group: `machine-${op}`,
          className: `phase-item phase-${op}`,
          style: `background-color: ${colors.bg}; border-color: ${colors.border}; color: ${colors.text};`,
          title: `${plan.order_number}\n${phase.name}: ${phase.duration_minutes} min\n${plan.product}`,
        })
      })
    })

    // Build machine sub-groups for operations that are actually used
    const machineGroupIds = []
    OPERATION_SEQUENCE.forEach((op, i) => {
      if (!usedOps.has(op)) return
      const gid = `machine-${op}`
      machineGroupIds.push(gid)
      const colors = PHASE_COLORS[op] || { border: '#9ca3af' }
      groups.push({
        id: gid,
        content: `<span class="machine-group-label"><span class="machine-pip" style="background:${colors.border}"></span>${MACHINE_LABELS[op] || op.toUpperCase()}</span>`,
        order: i + 1,
      })
    })

    // Update Production Line group to nest machine sub-groups
    if (machineGroupIds.length > 0) {
      groups[0] = {
        id: 'production-line',
        content: '<strong>Production Line</strong>',
        nestedGroups: machineGroupIds,
        showNested: true,
        order: 0,
      }
    }

    if (schedule.production_plans.length > 0) {
      const start = new Date(schedule.production_plans[0].starts_at)
      const end = new Date(schedule.production_plans[schedule.production_plans.length - 1].ends_at)
      
      let currentDay = new Date(start)
      currentDay.setHours(0, 0, 0, 0)
      
      while (currentDay <= end) {
        const nextDay = new Date(currentDay)
        nextDay.setDate(nextDay.getDate() + 1)
        
        items.push({
          id: `day-${currentDay.getTime()}`,
          content: '',
          start: new Date(currentDay),
          end: nextDay,
          type: 'background',
          className: 'day-boundary'
        })
        currentDay.setDate(currentDay.getDate() + 1)
      }
    }

    const options = {
      orientation: 'top',
      stack: false,
      showCurrentTime: true,
      margin: { item: 10, axis: 20 },
      editable: false,
      zoomMin: 1000 * 60 * 60 * 12,
      zoomMax: 1000 * 60 * 60 * 24 * 30,
      format: {
        minorLabels: {
          hour: 'HH:mm',
          day: 'D'
        },
        majorLabels: {
          day: 'MMM D',
          week: 'MMM D',
          month: 'MMMM YYYY'
        }
      },
      groupOrder: 'order'
    }

    if (timelineInstance.current) {
      timelineInstance.current.destroy()
    }

    timelineInstance.current = new window.vis.Timeline(
      timelineRef.current,
      items,
      groups,
      options
    )
  }

  const fetchSchedule = async () => {
    try {
      setLoading(true)
      const response = await axios.get('http://localhost:8000/api/scheduler/schedule')
      setSchedule(response.data)
      setError(null)
      // Refresh log after schedule recalculation (new version may have been created)
      fetchProductionLog()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchProductionLog = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/production-log')
      setProductionLog(response.data || [])
    } catch (err) {
      console.error('Failed to fetch production log:', err)
    }
  }

  const fetchProductionStates = async () => {
    try {
      setLoadingStates(true)
      const response = await axios.get('http://localhost:8000/api/production-state')
      console.log('Production states fetched:', response.data)
      setProductionStates(response.data.states || [])
    } catch (err) {
      console.error('Failed to fetch production states:', err)
    } finally {
      setLoadingStates(false)
    }
  }

  const loadOrderToLine = async (orderNumber) => {
    try {
      // Check if there's already an order on the line
      const activeOrder = productionStates.length > 0 ? productionStates[0] : null
      
      if (activeOrder && activeOrder.production_order !== orderNumber) {
        const confirmReplace = window.confirm(
          `Warning: ${activeOrder.production_order} is currently on the line.\n\n` +
          `Loading ${orderNumber} will remove ${activeOrder.production_order} from the line.\n\n` +
          `Continue?`
        )
        if (!confirmReplace) {
          return
        }
      }
      
      // Create production state entry (this will clear any existing order and set required steps)
      console.log(`Creating production state for ${orderNumber}...`)
      const createResponse = await axios.post(`http://localhost:8000/api/production-state/${orderNumber}`)
      console.log('Create state response:', createResponse.data)
      
      // Refresh production states
      await fetchProductionStates()
      autoAdvanceRef.current = {
        inFlight: false,
        lastAttemptKey: '',
        lastAttemptAt: 0,
        lastSuccessKey: '',
      }
      
      console.log('Order loaded successfully:', orderNumber)
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message
      console.error('Failed to load order:', errorMsg, err)
      alert(`Failed to load order: ${errorMsg}`)
    }
  }

  const completeOperation = async (orderNumber, operation, opts = { silent: false }) => {
    try {
      await axios.post(`http://localhost:8000/api/production-state/${orderNumber}/operation/${operation}/complete`)
      await fetchProductionStates()
      return true
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message
      console.error(`Failed to complete ${operation}:`, errorMsg)
      if (!opts.silent) {
        alert(`Cannot finish ${operation.toUpperCase()}: ${errorMsg}`)
      }
      return false
    }
  }

  const resetLoadedOrderFlow = async (orderNumber) => {
    if (!orderNumber) return
    try {
      await axios.post('http://localhost:8000/api/line/reset')
      await axios.post(`http://localhost:8000/api/production-state/${orderNumber}`)
      setLineVisionPhase('NOT_PRESENT')
      autoAdvanceRef.current = {
        inFlight: false,
        lastAttemptKey: '',
        lastAttemptAt: 0,
        lastSuccessKey: '',
      }
      await fetchProductionStates()
    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message
      console.error('Failed to reset flow:', errorMsg)
      alert(`Failed to reset flow: ${errorMsg}`)
    }
  }

  const getLoadedState = (states) =>
    states.find((state) =>
      OPERATION_SEQUENCE.some((op) => state[op]?.required) &&
      !OPERATION_SEQUENCE.every((op) => !state[op]?.required || state[op]?.finished_at)
    )

  const getNextRequiredOperation = (state) =>
    OPERATION_SEQUENCE.find((op) => state?.[op]?.required && !state?.[op]?.finished_at)

  useEffect(() => {
    if (activeTab !== 'line') return

    const loadedState = getLoadedState(productionStates)
    if (!loadedState) return

    const inferredOperation = PHASE_TO_OPERATION[lineVisionPhase]
    if (!inferredOperation) return

    const nextOperation = getNextRequiredOperation(loadedState)
    if (!nextOperation || inferredOperation !== nextOperation) return

    const key = `${loadedState.production_order}:${nextOperation}`
    const now = Date.now()
    const tracker = autoAdvanceRef.current

    if (tracker.inFlight) return
    if (tracker.lastSuccessKey === key) return
    if (tracker.lastAttemptKey === key && now - tracker.lastAttemptAt < 3000) return

    tracker.inFlight = true
    tracker.lastAttemptKey = key
    tracker.lastAttemptAt = now

    completeOperation(loadedState.production_order, nextOperation, { silent: true })
      .then((ok) => {
        if (ok) {
          tracker.lastSuccessKey = key
        }
      })
      .finally(() => {
        tracker.inFlight = false
      })
  }, [activeTab, productionStates, lineVisionPhase])
  const getPriorityColor = (priority) => {
    switch (priority) {
      case 1: return 'bg-red-100 text-red-800 border-red-300'
      case 2: return 'bg-orange-100 text-orange-800 border-orange-300'
      case 3: return 'bg-green-100 text-green-700 border-green-300'
      case 4: return 'bg-green-50 text-green-600 border-green-200'
      default: return 'bg-gray-100 text-gray-600 border-gray-300'
    }
  }

  const getPriorityLabel = (priority) => {
    switch (priority) {
      case 1: return 'High'
      case 2: return 'Medium'
      case 3: return 'Normal'
      case 4: return 'Low'
      default: return 'Very Low'
    }
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading production schedule...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md">
          <h2 className="text-red-800 font-semibold text-lg mb-2">Error Loading Schedule</h2>
          <p className="text-red-600">{error}</p>
          <button 
            onClick={fetchSchedule}
            className="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="w-full px-2 sm:px-3 lg:px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">ARKE Physical AI</h1>
              <p className="text-sm text-gray-500">Production Scheduling System - EDF Algorithm</p>
            </div>
            <button 
              onClick={fetchSchedule}
              disabled={loading}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>
          </div>
        </div>
      </div>

      {/* Only show tabs and content if we have data */}
      {!loading && schedule && (
        <>
          {/* Tabs */}
          <div className="w-full px-2 sm:px-3 lg:px-4 mt-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('gantt')}
              className={`${
                activeTab === 'gantt'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition`}
            >
              Gantt Chart
            </button>
            <button
              onClick={() => setActiveTab('schedule')}
              className={`${
                activeTab === 'schedule'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition`}
            >
              Production Schedule
            </button>
            <button
              onClick={() => setActiveTab('line')}
              className={`${
                activeTab === 'line'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition`}
            >
              Production Line
            </button>
            <button
              onClick={() => setActiveTab('log')}
              className={`${
                activeTab === 'log'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition`}
            >
              Schedule Log
            </button>
            <button
              onClick={() => setActiveTab('live-view')}
              className={`${
                activeTab === 'live-view'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition`}
            >
              Live View
            </button>
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="w-full px-2 sm:px-3 lg:px-4 py-6">
        {activeTab === 'gantt' && (
          <div className="space-y-4">
            {/* Header */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-slate-800 flex items-center justify-center">
                  <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h7" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-base font-semibold text-gray-900">Production Gantt</h2>
                  <p className="text-xs text-gray-400">Single-line sequential processing timeline</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-gray-500 bg-gray-50 border border-gray-200 rounded-md px-2.5 py-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#fef2f2] border border-[#f87171]"></span>
                  P1
                </span>
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-gray-500 bg-gray-50 border border-gray-200 rounded-md px-2.5 py-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#fff7ed] border border-[#fb923c]"></span>
                  P2
                </span>
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-gray-500 bg-gray-50 border border-gray-200 rounded-md px-2.5 py-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#ecfdf5] border border-[#34d399]"></span>
                  P3/P4
                </span>
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-gray-500 bg-gray-50 border border-gray-200 rounded-md px-2.5 py-1">
                  <span className="w-2.5 h-2.5 rounded-sm border border-red-300" style={{background:'repeating-linear-gradient(45deg,#fef2f2,#fef2f2 3px,#fecaca 3px,#fecaca 6px)'}}></span>
                  Late
                </span>
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-gray-500 bg-gray-50 border border-gray-200 rounded-md px-2.5 py-1">
                  <span className="w-0.5 h-3 bg-red-500 rounded-full"></span>
                  Now
                </span>
                {schedule?.production_plans?.length > 0 && (
                  <span className="text-[11px] font-medium text-gray-400 bg-gray-50 rounded-md px-2.5 py-1">
                    {schedule.production_plans.length} orders
                  </span>
                )}
              </div>
            </div>
            {/* Timeline container */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
              <div ref={timelineRef} className="w-full gantt-container"></div>
            </div>
          </div>
        )}

        {activeTab === 'schedule' && (
          <div className="space-y-4">
            {/* Stats + Push button */}
            <div className="flex items-center justify-between mb-2">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 flex-1">
                <div className="bg-white rounded-lg shadow p-4 border">
                  <div className="text-sm text-gray-500">Total Orders</div>
                  <div className="text-2xl font-bold text-gray-900">{schedule?.production_plans?.length || 0}</div>
                </div>
                <div className="bg-white rounded-lg shadow p-4 border">
                  <div className="text-sm text-gray-500">Policy</div>
                  <div className="text-2xl font-bold text-blue-600">EDF</div>
                </div>
                <div className="bg-white rounded-lg shadow p-4 border">
                  <div className="text-sm text-gray-500">Conflicts Detected</div>
                  <div className="text-2xl font-bold text-red-600">{schedule?.conflicts?.length || 0}</div>
                </div>
                <div className="bg-white rounded-lg shadow p-4 border">
                  <div className="text-sm text-gray-500">Pushed to Arke</div>
                  <div className="text-2xl font-bold text-green-600">
                    {Object.values(pushStatus).filter(r => r.status === 'pushed').length}
                    <span className="text-sm font-normal text-gray-400">
                      {' / '}{schedule?.production_plans?.length || 0}
                    </span>
                  </div>
                </div>
              </div>
              <button
                onClick={pushToArke}
                disabled={pushing}
                className={`ml-4 px-5 py-3 rounded-lg font-semibold text-sm shadow transition flex items-center gap-2 ${
                  pushing
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                }`}
              >
                {pushing ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Pushing…
                  </>
                ) : (
                  <>
                    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5-5m0 0l5 5m-5-5v12" />
                    </svg>
                    Push to Arke
                  </>
                )}
              </button>
            </div>

            {/* Production Plans */}
            <div className="space-y-3">
              {schedule?.production_plans?.map((plan, index) => {
                const ps = pushStatus[plan.order_number]
                return (
                <div key={index} className="bg-white rounded-lg shadow-sm border hover:shadow-md transition">
                  <div className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="text-lg font-semibold text-gray-900">#{index + 1}</span>
                          <h3 className="text-lg font-semibold text-gray-900">{plan.order_number}</h3>
                          <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getPriorityColor(plan.priority)}`}>
                            P{plan.priority} - {getPriorityLabel(plan.priority)}
                          </span>
                          {/* Push status badge */}
                          {ps?.status === 'pushed' && (
                            <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700 border border-green-200">
                              ✓ Pushed
                            </span>
                          )}
                          {ps?.status === 'failed' && (
                            <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700 border border-red-200" title={ps.error_message}>
                              ✗ Failed
                            </span>
                          )}
                          {ps?.status === 'pushing' && (
                            <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700 border border-yellow-200 animate-pulse">
                              ⏳ Pushing…
                            </span>
                          )}
                          {!ps && (
                            <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-500 border border-gray-200">
                              ○ Pending
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Product:</span>
                            <p className="font-medium text-gray-900">{plan.product}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Customer:</span>
                            <p className="font-medium text-gray-900">{plan.customer}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Quantity:</span>
                            <p className="font-medium text-gray-900">{plan.quantity} units</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Schedule:</span>
                            <p className="font-medium text-gray-900">
                              {formatDate(plan.starts_at)} → {formatDate(plan.ends_at)}
                            </p>
                          </div>
                        </div>
                        {plan.reasoning && (
                          <div className="mt-3 p-3 bg-gray-50 rounded-md">
                            <p className="text-sm text-gray-700">
                              <span className="font-medium">Reasoning:</span> {plan.reasoning}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
                )
              })}
            </div>
          </div>
        )}

        {activeTab === 'line' && (
          <div className="flex items-stretch gap-4" style={{ minHeight: 'calc(100vh - 170px)' }}>
            {/* ── Sidebar: Upcoming Orders ── */}
            <div className="w-80 flex-shrink-0">
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden sticky top-6 h-[calc(100vh-170px)] flex flex-col">
                <div className="px-5 py-4 border-b border-gray-100 bg-gray-50">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">Upcoming Orders</h2>
                    <span className="text-xs font-medium text-gray-500 bg-gray-200 rounded-full px-2 py-0.5">
                      {schedule?.production_plans?.length || 0}
                    </span>
                  </div>
                </div>
                  <div className="divide-y divide-gray-100 flex-1 min-h-0 overflow-y-auto">
                  {schedule?.production_plans?.map((plan, index) => {
                    const state = productionStates.find(s => s.production_order === plan.order_number)
                    const isLoaded = state && ['smt', 'reflow', 'tht', 'aoi', 'test', 'coating', 'pack'].some(op => state[op]?.required)
                      && !['smt', 'reflow', 'tht', 'aoi', 'test', 'coating', 'pack'].every(op => !state[op]?.required || state[op]?.finished_at)
                    const deadline = new Date(plan.deadline)
                    const isLate = new Date(plan.ends_at) > deadline

                    return (
                      <div key={plan.order_number} className={`px-5 py-3 hover:bg-gray-50 transition cursor-default ${isLoaded ? 'bg-blue-50 border-l-4 border-l-blue-500' : ''}`}>
                        <div className="flex items-center justify-between mb-1">
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-gray-400 font-mono">#{index + 1}</span>
                            <span className="font-semibold text-sm text-gray-900">{plan.order_number}</span>
                          </div>
                          {isLoaded && (
                            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" title="On Line"></span>
                          )}
                        </div>
                        <div className="text-xs text-gray-500 mb-1.5">{plan.customer} · {plan.quantity}× {plan.product}</div>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5">
                            <span className={`inline-block w-1.5 h-1.5 rounded-full ${plan.priority <= 2 ? 'bg-red-400' : 'bg-green-400'}`}></span>
                            <span className="text-xs text-gray-400">P{plan.priority}</span>
                            <span className={`text-xs ${isLate ? 'text-red-500 font-medium' : 'text-gray-400'}`}>
                              · {deadline.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                            </span>
                          </div>
                          {!isLoaded && (
                            <button
                              onClick={() => loadOrderToLine(plan.order_number)}
                              className="text-xs font-medium text-blue-600 hover:text-blue-800 transition"
                            >
                              Load →
                            </button>
                          )}
                        </div>
                      </div>
                    )
                  })}
                  {(!schedule?.production_plans || schedule.production_plans.length === 0) && (
                    <div className="px-5 py-8 text-center text-sm text-gray-400">No orders scheduled</div>
                  )}
                </div>
              </div>
            </div>

            {/* ── Main Area: Loaded Order ── */}
            <div className="flex-1 min-w-0">
              {(() => {
                const loadedState = productionStates.find(s =>
                  ['smt', 'reflow', 'tht', 'aoi', 'test', 'coating', 'pack'].some(op => s[op]?.required)
                  && !['smt', 'reflow', 'tht', 'aoi', 'test', 'coating', 'pack'].every(op => !s[op]?.required || s[op]?.finished_at)
                )
                const loadedPlan = loadedState && schedule?.production_plans?.find(p => p.order_number === loadedState.production_order)

                if (!loadedState) {
                  return (
                    <div className="h-full flex items-center justify-center">
                      <div className="text-center py-20">
                        <div className="mx-auto w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                          <svg className="h-8 w-8 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                          </svg>
                        </div>
                        <h3 className="text-lg font-semibold text-gray-700 mb-1">No Order on Line</h3>
                        <p className="text-sm text-gray-400">Select an order from the sidebar to load it onto the production line</p>
                      </div>
                    </div>
                  )
                }

                const ops = [
                  { key: 'smt', label: 'SMT', desc: 'Surface Mount Technology' },
                  { key: 'reflow', label: 'Reflow', desc: 'Reflow Soldering' },
                  { key: 'tht', label: 'THT', desc: 'Through-Hole Technology' },
                  { key: 'aoi', label: 'AOI', desc: 'Automated Optical Inspection' },
                  { key: 'test', label: 'Test', desc: 'Functional Testing' },
                  { key: 'coating', label: 'Coating', desc: 'Conformal Coating' },
                  { key: 'pack', label: 'Pack', desc: 'Final Packaging' },
                ]
                const phaseFlow = [
                  { op: 'smt', label: 'SMT' },
                  { op: 'reflow', label: 'Reflow' },
                  { op: 'tht', label: 'THT' },
                  { op: 'aoi', label: 'AOI' },
                  { op: 'test', label: 'Test' },
                  { op: 'coating', label: 'Coating' },
                  { op: 'pack', label: 'Pack' },
                ]
                const completedOps = ops.filter(o => loadedState[o.key]?.required && loadedState[o.key]?.finished_at).length
                const totalRequired = ops.filter(o => loadedState[o.key]?.required).length
                const progressPct = totalRequired > 0 ? Math.round((completedOps / totalRequired) * 100) : 0
                const currentOperation = PHASE_TO_OPERATION[lineVisionPhase]

                return (
                  <div className="space-y-4">
                    {/* Order Header Card */}
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                      <div className="bg-gradient-to-r from-slate-800 to-slate-700 px-6 py-5 text-white">
                        <div className="flex items-start justify-between">
                          <div>
                            <div className="flex items-center gap-3 mb-1">
                              <span className="text-xs font-medium uppercase tracking-wider text-slate-300">Active Order</span>
                              <span className="inline-flex items-center gap-1 text-xs font-medium bg-green-500/20 text-green-300 rounded-full px-2 py-0.5">
                                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span>
                                Live
                              </span>
                              <button
                                onClick={() => resetLoadedOrderFlow(loadedState.production_order)}
                                className="ml-2 inline-flex items-center rounded-full border border-amber-300/50 bg-amber-400/15 px-3 py-1 text-xs font-medium text-amber-100 hover:bg-amber-400/25"
                              >
                                Reset Flow
                              </button>
                            </div>
                            <h2 className="text-2xl font-bold tracking-tight">{loadedState.production_order}</h2>
                            {loadedPlan && (
                              <p className="text-sm text-slate-300 mt-1">{loadedPlan.customer} · {loadedPlan.quantity}× {loadedPlan.product}</p>
                            )}
                          </div>
                          <div className="text-right">
                            <div className="text-3xl font-bold">{progressPct}%</div>
                            <div className="text-xs text-slate-400">{completedOps} of {totalRequired} ops</div>
                          </div>
                        </div>
                        {/* Progress bar */}
                        <div className="mt-4 h-2 bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-green-400 to-emerald-400 rounded-full transition-all duration-500"
                            style={{ width: `${progressPct}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="px-4 py-3 bg-slate-900 border-t border-slate-700">
                        <div className="grid grid-cols-7 gap-2">
                          {phaseFlow.map((phase) => {
                            const finished = loadedState[phase.op]?.required && loadedState[phase.op]?.finished_at
                            const active = currentOperation === phase.op
                            return (
                              <div
                                key={phase.op}
                                className={`rounded-lg border px-2 py-1.5 text-center text-xs font-medium transition ${
                                  finished
                                    ? 'bg-emerald-500/20 border-emerald-400/50 text-emerald-200'
                                    : active
                                      ? 'bg-blue-500/25 border-blue-300/60 text-blue-100'
                                      : 'bg-slate-800 border-slate-600 text-slate-300'
                                }`}
                              >
                                {phase.label}
                              </div>
                            )
                          })}
                        </div>
                      </div>

                      {/* Order details row */}
                      {loadedPlan && (
                        <div className="px-6 py-3 bg-gray-50 border-t border-gray-100 flex divide-x divide-gray-200 text-sm">
                          <div className="pr-6">
                            <span className="text-gray-400">Priority</span>
                            <span className={`ml-2 font-semibold ${loadedPlan.priority <= 2 ? 'text-red-600' : 'text-gray-800'}`}>
                              P{loadedPlan.priority} – {getPriorityLabel(loadedPlan.priority)}
                            </span>
                          </div>
                          <div className="px-6">
                            <span className="text-gray-400">Deadline</span>
                            <span className="ml-2 font-semibold text-gray-800">
                              {new Date(loadedPlan.deadline).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                            </span>
                          </div>
                          <div className="px-6">
                            <span className="text-gray-400">Scheduled</span>
                            <span className="ml-2 font-semibold text-gray-800">
                              {formatDate(loadedPlan.starts_at)} → {formatDate(loadedPlan.ends_at)}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>

                    <LiveViewInlineSection onPhaseChange={setLineVisionPhase} />
                  </div>
                )
              })()}
            </div>
          </div>
        )}

        {activeTab === 'log' && (
          <ScheduleLogTab
            schedule={schedule}
            productionLog={productionLog}
            onRefresh={fetchProductionLog}
          />
        )}

        {activeTab === 'live-view' && (
          <LiveViewDashboard />
        )}
      </div>
        </>
      )}
    </div>
  )
}

export default App
