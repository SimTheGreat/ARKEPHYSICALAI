import { useState, useEffect } from 'react'
import axios from 'axios'

function App() {
  const [schedule, setSchedule] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('schedule')

  useEffect(() => {
    fetchSchedule()
  }, [])

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
      case 1: return 'bg-red-100 text-red-800 border-red-200'
      case 2: return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      case 3: return 'bg-blue-100 text-blue-800 border-blue-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
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
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
            >
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
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
                            P{plan.priority}
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

        {activeTab === 'conflicts' && (
          <div className="space-y-4">
            {schedule?.conflicts?.length > 0 ? (
              schedule.conflicts.map((conflict, index) => (
                <div key={index} className="bg-white rounded-lg shadow-sm border-l-4 border-red-500 p-6">
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      <svg className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    </div>
                    <div className="ml-3 flex-1">
                      <h3 className="text-lg font-medium text-gray-900 mb-2">
                        Priority vs Deadline Conflict
                      </h3>
                      <div className="space-y-3">
                        <div className="bg-yellow-50 p-3 rounded">
                          <p className="text-sm font-medium text-yellow-800">Priority-First Approach:</p>
                          <p className="text-sm text-yellow-700 mt-1">
                            {conflict.priority_first.order} (P{conflict.priority_first.priority}, Deadline: {conflict.priority_first.deadline})
                          </p>
                        </div>
                        <div className="bg-green-50 p-3 rounded">
                          <p className="text-sm font-medium text-green-800">EDF Approach (Recommended):</p>
                          <p className="text-sm text-green-700 mt-1">
                            {conflict.edf_first.order} (P{conflict.edf_first.priority}, Deadline: {conflict.edf_first.deadline})
                          </p>
                        </div>
                        <div className="bg-blue-50 p-3 rounded">
                          <p className="text-sm text-blue-900">
                            <span className="font-medium">Resolution:</span> {conflict.resolution}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="bg-green-50 rounded-lg p-8 text-center">
                <svg className="mx-auto h-12 w-12 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h3 className="mt-2 text-lg font-medium text-green-900">No Conflicts Detected</h3>
                <p className="mt-1 text-sm text-green-700">All production orders can be completed on schedule.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
