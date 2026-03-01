import { useEffect, useRef } from 'react'

export default function GanttTab({ schedule }) {
  const timelineRef = useRef(null)
  const timelineInstance = useRef(null)

  useEffect(() => {
    if (schedule && timelineRef.current && window.vis) {
      initTimeline()
    }
    return () => {
      if (timelineInstance.current) {
        timelineInstance.current.destroy()
        timelineInstance.current = null
      }
    }
  }, [schedule])

  const initTimeline = () => {
    if (!schedule || !timelineRef.current) return

    const items = []
    const groups = [{ id: 'production-line', content: 'Production Line' }]

    schedule.production_plans.forEach((plan, index) => {
      const startDate = new Date(plan.starts_at)
      const endDate = new Date(plan.ends_at)
      const deadline = new Date(plan.deadline)

      const deltaDays = Math.ceil((deadline - endDate) / (1000 * 60 * 60 * 24))
      const isLate = deltaDays < 0
      const deadlineText = `Due: ${deadline.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`

      items.push({
        id: `order-${index}`,
        content: `<div class="order-label">
          <strong>${plan.order_number}</strong>
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
        minorLabels: { hour: 'HH:mm', day: 'D' },
        majorLabels: { day: 'MMM D', week: 'MMM D', month: 'MMMM YYYY' }
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

  return (
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
  )
}
