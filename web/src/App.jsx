import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

function App() {
  const [schedule, setSchedule] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('gantt')
  const timelineRef = useRef(null)
  const timelineInstance = useRef(null)

  useEffect(() => {
    fetchSchedule()
  }, [])

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
      groupOrder: 'id'
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
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

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
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
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
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
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
              onClick={() => setActiveTab('conflicts')}
              className={`${
                activeTab === 'conflicts'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm transition`}
            >
              Conflicts
              {schedule?.conflicts?.length > 0 && (
                <span className="ml-2 bg-red-100 text-red-600 py-0.5 px-2 rounded-full text-xs">
                  {schedule.conflicts.length}
                </span>
              )}
            </button>
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'gantt' && (
          <div className="space-y-4">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Production Timeline - Single Line Sequential Processing</h2>
              <div ref={timelineRef} className="w-full" style={{ height: '400px' }}></div>
            </div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-800">
                <strong>Timeline View:</strong> Production orders run sequentially on a single production line. 
                Day boundaries shown with vertical lines (480 min/day working time). 
                Hatched red bars show orders running late. Delta shows days until/past deadline.
                ⚠ indicates SO-003/SO-017 EDF vs Priority conflict.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'schedule' && (
          <div className="space-y-4">
            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
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
            </div>

            {/* Production Plans */}
            <div className="space-y-3">
              {schedule?.production_plans?.map((plan, index) => (
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
              ))}
            </div>
          </div>
        )}

        {activeTab === 'log' && (
          <div className="space-y-4">
            {/* Success Banner */}
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center gap-2">
                <svg className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="font-semibold text-green-900">Scheduling Completed Successfully</span>
              </div>
              <p className="text-sm text-green-700 mt-1 ml-7">
                {schedule?.production_plans?.length || 0} orders scheduled using EDF (Earliest Deadline First) policy
              </p>
            </div>

            {/* Event Log */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900">Scheduling Events</h2>
                <p className="text-sm text-gray-500 mt-1">Chronological log of scheduling decisions</p>
              </div>
              
              <div className="divide-y divide-gray-100">
                {schedule?.production_plans?.map((plan, index) => {
                  // Check if this order is involved in a priority override
                  const isOverride = schedule?.conflicts?.some(c => 
                    c.priority_first.order === plan.order_number || 
                    c.edf_first.order === plan.order_number
                  )
                  const overrideInfo = schedule?.conflicts?.find(c => 
                    c.edf_first.order === plan.order_number
                  )
                  
                  return (
                    <div key={index} className={`px-6 py-4 ${
                      isOverride ? 'bg-yellow-50' : 'hover:bg-gray-50'
                    } transition`}>
                      <div className="flex items-start gap-4">
                        {/* Event Number */}
                        <div className="flex-shrink-0">
                          <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-700 font-semibold text-sm">
                            {index + 1}
                          </span>
                        </div>
                        
                        {/* Event Details */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="font-semibold text-gray-900">{plan.order_number}</h3>
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${getPriorityColor(plan.priority)}`}>
                              P{plan.priority} - {getPriorityLabel(plan.priority)}
                            </span>
                          </div>
                          
                          <div className="text-sm text-gray-600 space-y-1">
                            <div className="flex items-center gap-2">
                              <span className="text-gray-500">Customer:</span>
                              <span className="font-medium">{plan.customer}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-gray-500">Product:</span>
                              <span className="font-medium">{plan.quantity}x {plan.product}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-gray-500">Scheduled:</span>
                              <span className="font-medium">{formatDate(plan.starts_at)} → {formatDate(plan.ends_at)}</span>
                            </div>
                          </div>

                          {/* EDF Reasoning */}
                          <div className="mt-2 text-sm">
                            <span className="text-gray-700">{plan.reasoning}</span>
                          </div>

                          {/* Priority Override Notice */}
                          {overrideInfo && (
                            <div className="mt-3 bg-yellow-100 border border-yellow-300 rounded-lg p-3">
                              <div className="flex items-start gap-2">
                                <svg className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                <div className="text-sm">
                                  <p className="font-semibold text-yellow-900">EDF Policy Override</p>
                                  <p className="text-yellow-800 mt-1">{overrideInfo.resolution}</p>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Summary Footer */}
              <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Total Orders:</span>
                    <span className="ml-2 font-semibold text-gray-900">{schedule?.production_plans?.length || 0}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Timespan:</span>
                    <span className="ml-2 font-semibold text-gray-900">
                      {schedule?.production_plans?.length > 0 && 
                        formatDate(schedule.production_plans[0].starts_at) + ' - ' + 
                        formatDate(schedule.production_plans[schedule.production_plans.length - 1].ends_at)
                      }
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-500">Policy:</span>
                    <span className="ml-2 font-semibold text-blue-600">EDF</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'conflicts' && (
          <div className="space-y-6">
            {/* Deadline Miss Conflicts (Critical) */}
            {schedule?.production_plans?.some(p => new Date(p.ends_at) > new Date(p.deadline)) && (
              <div className="space-y-4">
                {schedule.production_plans
                  .filter(p => new Date(p.ends_at) > new Date(p.deadline))
                  .map((plan, index) => {
                    const endDate = new Date(plan.ends_at)
                    const deadline = new Date(plan.deadline)
                    const daysLate = Math.ceil((endDate - deadline) / (1000 * 60 * 60 * 24))
                    
                    return (
                      <div key={index} className="bg-red-50 rounded-lg shadow-sm border-l-4 border-red-600 p-6">
                        <div className="flex items-start">
                          <div className="flex-shrink-0">
                            <svg className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          </div>
                          <div className="ml-3 flex-1">
                            <h3 className="text-lg font-semibold text-red-900 mb-2">
                              {plan.order_number} - Cannot Meet Deadline
                            </h3>
                            <div className="space-y-2">
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <span className="text-red-700 font-medium">Product:</span>
                                  <p className="text-red-900">{plan.product}</p>
                                </div>
                                <div>
                                  <span className="text-red-700 font-medium">Customer:</span>
                                  <p className="text-red-900">{plan.customer}</p>
                                </div>
                                <div>
                                  <span className="text-red-700 font-medium">Scheduled Completion:</span>
                                  <p className="text-red-900">{formatDate(plan.ends_at)}</p>
                                </div>
                                <div>
                                  <span className="text-red-700 font-medium">Deadline:</span>
                                  <p className="text-red-900">{formatDate(plan.deadline)}</p>
                                </div>
                              </div>
                              <div className="bg-red-100 p-3 rounded mt-3">
                                <p className="text-sm text-red-900">
                                  <span className="font-bold">⚠️ {daysLate} day{daysLate > 1 ? 's' : ''} late</span> - 
                                  This order will miss its deadline due to production capacity constraints. 
                                  Consider expediting, splitting batches, or communicating delay to customer.
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )
                  })}
              </div>
            )}

            {/* Priority vs Deadline Conflicts (Informational) */}
            {schedule?.conflicts?.length > 0 && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-4">
                  <svg className="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h2 className="text-xl font-bold text-yellow-900">EDF Policy Overruled Priority Level</h2>
                </div>
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                  <p className="text-sm text-yellow-800">
                    The following orders were scheduled based on <strong>deadline</strong> rather than <strong>priority level</strong>. 
                    EDF policy prioritizes meeting deadlines over respecting priority order.
                    <strong> All deadlines are still met.</strong>
                  </p>
                </div>
                
                {schedule.conflicts.map((conflict, index) => {
                  // Find the full order details from production_plans
                  const highPriorityOrder = schedule.production_plans.find(p => p.order_number === conflict.priority_first.order)
                  const earlierDeadlineOrder = schedule.production_plans.find(p => p.order_number === conflict.edf_first.order)
                  
                  return (
                    <div key={index} className="bg-white rounded-lg shadow-sm border-l-4 border-yellow-500 p-6">
                      <div className="mb-4">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2 flex items-center gap-2">
                          <span className="text-yellow-600">⚠️</span>
                          Conflict #{index + 1}: Priority vs Deadline
                        </h3>
                        <p className="text-sm text-gray-600">
                          EDF policy overruled priority level to meet earlier deadline
                        </p>
                      </div>

                      <div className="grid md:grid-cols-2 gap-4 mb-4">
                        {/* High Priority Order - Would Go First by Priority */}
                        <div className="bg-red-50 border-2 border-red-200 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <span className="bg-red-600 text-white text-xs font-bold px-2 py-1 rounded">
                              P{conflict.priority_first.priority} - MORE CRITICAL
                            </span>
                            <span className="text-xs text-red-700">But later deadline</span>
                          </div>
                          <div className="space-y-2">
                            <div>
                              <p className="text-sm font-bold text-red-900">{conflict.priority_first.order}</p>
                              {highPriorityOrder && (
                                <>
                                  <p className="text-xs text-red-700">{highPriorityOrder.customer}</p>
                                  <p className="text-xs text-red-700">{highPriorityOrder.quantity}x {highPriorityOrder.product}</p>
                                </>
                              )}
                            </div>
                            <div className="border-t border-red-200 pt-2 space-y-1">
                              <div className="flex justify-between text-xs">
                                <span className="text-red-700">Priority:</span>
                                <span className="font-bold text-red-900">P{conflict.priority_first.priority} - {getPriorityLabel(conflict.priority_first.priority)}</span>
                              </div>
                              <div className="flex justify-between text-xs">
                                <span className="text-red-700">Deadline:</span>
                                <span className="font-bold text-red-900">{conflict.priority_first.deadline}</span>
                              </div>
                              {highPriorityOrder && (
                                <>
                                  <div className="flex justify-between text-xs">
                                    <span className="text-red-700">Scheduled:</span>
                                    <span className="font-medium text-red-800">{formatDate(highPriorityOrder.starts_at)}</span>
                                  </div>
                                  <div className="flex justify-between text-xs">
                                    <span className="text-red-700">Completes:</span>
                                    <span className="font-medium text-red-800">{formatDate(highPriorityOrder.ends_at)}</span>
                                  </div>
                                </>
                              )}
                            </div>
                          </div>
                          <div className="mt-3 bg-red-100 rounded p-2">
                            <p className="text-xs text-red-800">
                              <strong>Position Impact:</strong> Moved back in queue to prioritize earlier deadline
                            </p>
                          </div>
                        </div>

                        {/* Earlier Deadline Order - Goes First by EDF */}
                        <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
                          <div className="flex items-center gap-2 mb-3">
                            <span className="bg-green-600 text-white text-xs font-bold px-2 py-1 rounded">
                              P{conflict.edf_first.priority} - EARLIER DEADLINE
                            </span>
                            <span className="text-xs text-green-700">Lower priority but goes first</span>
                          </div>
                          <div className="space-y-2">
                            <div>
                              <p className="text-sm font-bold text-green-900">{conflict.edf_first.order}</p>
                              {earlierDeadlineOrder && (
                                <>
                                  <p className="text-xs text-green-700">{earlierDeadlineOrder.customer}</p>
                                  <p className="text-xs text-green-700">{earlierDeadlineOrder.quantity}x {earlierDeadlineOrder.product}</p>
                                </>
                              )}
                            </div>
                            <div className="border-t border-green-200 pt-2 space-y-1">
                              <div className="flex justify-between text-xs">
                                <span className="text-green-700">Priority:</span>
                                <span className="font-bold text-green-900">P{conflict.edf_first.priority} - {getPriorityLabel(conflict.edf_first.priority)}</span>
                              </div>
                              <div className="flex justify-between text-xs">
                                <span className="text-green-700">Deadline:</span>
                                <span className="font-bold text-green-900">{conflict.edf_first.deadline}</span>
                              </div>
                              {earlierDeadlineOrder && (
                                <>
                                  <div className="flex justify-between text-xs">
                                    <span className="text-green-700">Scheduled:</span>
                                    <span className="font-medium text-green-800">{formatDate(earlierDeadlineOrder.starts_at)}</span>
                                  </div>
                                  <div className="flex justify-between text-xs">
                                    <span className="text-green-700">Completes:</span>
                                    <span className="font-medium text-green-800">{formatDate(earlierDeadlineOrder.ends_at)}</span>
                                  </div>
                                </>
                              )}
                            </div>
                          </div>
                          <div className="mt-3 bg-green-100 rounded p-2">
                            <p className="text-xs text-green-800">
                              <strong>Position Impact:</strong> Moved ahead to meet earlier deadline ✓
                            </p>
                          </div>
                        </div>
                      </div>

                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <p className="text-sm text-blue-900">
                          <span className="font-semibold">EDF Decision:</span> {conflict.resolution}
                        </p>
                        <div className="mt-2 flex items-center gap-2 text-xs text-blue-700">
                          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span>Both orders meet their deadlines with this scheduling</span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {/* No Conflicts */}
            {!schedule?.production_plans?.some(p => new Date(p.ends_at) > new Date(p.deadline)) && 
             schedule?.conflicts?.length === 0 && (
              <div className="bg-green-50 rounded-lg p-8 text-center">
                <svg className="mx-auto h-12 w-12 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h3 className="mt-2 text-lg font-medium text-green-900">Perfect Schedule</h3>
                <p className="mt-1 text-sm text-green-700">
                  All production orders can be completed on schedule with no conflicts.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
        </>
      )}
    </div>
  )
}

export default App
