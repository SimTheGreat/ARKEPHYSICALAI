import { getPriorityLabel, formatDate } from './tabHelpers'

export default function ConflictsTab({ schedule }) {
  return (
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
  )
}
