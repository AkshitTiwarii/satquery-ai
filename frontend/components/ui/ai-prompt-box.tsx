'use client'

import React from 'react'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { ArrowUp, Paperclip, Square, X, StopCircle, Mic, Globe, BrainCog, FolderCode } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '@/lib/utils'

// Embedded CSS for minimal custom styles (SSR safe)
const styles = `
  *:focus-visible {
    outline-offset: 0 !important;
    --ring-offset: 0 !important;
  }
  textarea::-webkit-scrollbar {
    width: 6px;
  }
  textarea::-webkit-scrollbar-track {
    background: transparent;
  }
  textarea::-webkit-scrollbar-thumb {
    background-color: #334155;
    border-radius: 3px;
  }
  textarea::-webkit-scrollbar-thumb:hover {
    background-color: #475569;
  }
`

if (typeof document !== 'undefined') {
  const styleId = 'prompt-input-styles'
  if (!document.getElementById(styleId)) {
    const styleSheet = document.createElement('style')
    styleSheet.id = styleId
    styleSheet.innerText = styles
    document.head.appendChild(styleSheet)
  }
}

// Textarea Component
export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  className?: string
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(({ className, ...props }, ref) => (
  <textarea
    className={cn(
      'flex w-full rounded-md border-none bg-transparent px-3 py-2.5 text-base text-gray-100 placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-0 disabled:cursor-not-allowed disabled:opacity-50 min-h-[44px] resize-none scrollbar-thin scrollbar-thumb-[#334155] scrollbar-track-transparent hover:scrollbar-thumb-[#475569]',
      className
    )}
    ref={ref}
    rows={1}
    {...props}
  />
))
Textarea.displayName = 'Textarea'

// Tooltip Components
export const TooltipProvider = TooltipPrimitive.Provider
export const Tooltip = TooltipPrimitive.Root
export const TooltipTrigger = TooltipPrimitive.Trigger
export const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      'z-50 overflow-hidden rounded-md border border-white/15 bg-[#0a1017] px-3 py-1.5 text-xs text-white shadow-xl animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
      className
    )}
    {...props}
  />
))
TooltipContent.displayName = TooltipPrimitive.Content.displayName

// Dialog Components
export const Dialog = DialogPrimitive.Root
export const DialogPortal = DialogPrimitive.Portal
export const DialogOverlay = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Overlay
    ref={ref}
    className={cn(
      'fixed inset-0 z-50 bg-black/75 backdrop-blur-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
      className
    )}
    {...props}
  />
))
DialogOverlay.displayName = DialogPrimitive.Overlay.displayName

export const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        'fixed left-[50%] top-[50%] z-50 grid w-full max-w-[90vw] md:max-w-[800px] translate-x-[-50%] translate-y-[-50%] gap-4 border border-white/15 bg-[#0a1017] p-0 shadow-2xl duration-300 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 rounded-2xl',
        className
      )}
      {...props}
    >
      {children}
      <DialogPrimitive.Close className="absolute right-4 top-4 z-10 rounded-full bg-white/10 p-2 text-white/70 hover:bg-white/20 hover:text-white transition-all">
        <X className="h-5 w-5" />
        <span className="sr-only">Close</span>
      </DialogPrimitive.Close>
    </DialogPrimitive.Content>
  </DialogPortal>
))
DialogContent.displayName = DialogPrimitive.Content.displayName

export const DialogTitle = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Title>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Title
    ref={ref}
    className={cn('text-lg font-semibold leading-none tracking-tight text-gray-100', className)}
    {...props}
  />
))
DialogTitle.displayName = DialogPrimitive.Title.displayName

// Button Component
export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost'
  size?: 'default' | 'sm' | 'lg' | 'icon'
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    const variantClasses = {
      default: 'bg-white hover:bg-white/90 text-black shadow-md',
      outline: 'border border-white/20 bg-transparent hover:bg-white/10 text-white',
      ghost: 'bg-transparent hover:bg-white/10 text-white/80 hover:text-white',
    }
    const sizeClasses = {
      default: 'h-10 px-4 py-2 text-sm',
      sm: 'h-8 px-3 text-xs',
      lg: 'h-12 px-6 text-base',
      icon: 'h-8 w-8 rounded-full aspect-[1/1]',
    }
    return (
      <button
        className={cn(
          'inline-flex items-center justify-center font-medium transition-colors focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50 cursor-pointer',
          variantClasses[variant],
          sizeClasses[size],
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

// VoiceRecorder Component
export interface VoiceRecorderProps {
  isRecording: boolean
  onStartRecording: () => void
  onStopRecording: (duration: number) => void
  visualizerBars?: number
}

export const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  isRecording,
  onStartRecording,
  onStopRecording,
  visualizerBars = 32,
}) => {
  const [time, setTime] = React.useState(0)
  const timerRef = React.useRef<NodeJS.Timeout | null>(null)

  React.useEffect(() => {
    if (isRecording) {
      onStartRecording()
      timerRef.current = setInterval(() => setTime((t) => t + 1), 1000)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      onStopRecording(time)
      setTime(0)
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isRecording, onStartRecording, onStopRecording, time])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center w-full transition-all duration-300 py-3',
        isRecording ? 'opacity-100' : 'opacity-0 h-0 overflow-hidden'
      )}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
        <span className="font-mono text-xs text-white/80">{formatTime(time)}</span>
        <span className="font-mono text-[10px] text-white/40 uppercase tracking-wider">Listening to query</span>
      </div>
      <div className="w-full h-10 flex items-center justify-center gap-1 px-4">
        {[...Array(visualizerBars)].map((_, i) => (
          <div
            key={i}
            className="w-1 rounded-full bg-gradient-to-t from-[#8eb7ff] to-white/90 animate-pulse"
            style={{
              height: `${Math.max(15, ((Math.sin(i * 0.4 + time) + 1) / 2) * 90 + Math.random() * 10)}%`,
              animationDelay: `${i * 0.04}s`,
              animationDuration: `${0.4 + (i % 5) * 0.1}s`,
            }}
          />
        ))}
      </div>
    </div>
  )
}

// ImageViewDialog Component
export interface ImageViewDialogProps {
  imageUrl: string | null
  onClose: () => void
}

export const ImageViewDialog: React.FC<ImageViewDialogProps> = ({ imageUrl, onClose }) => {
  if (!imageUrl) return null
  return (
    <Dialog open={!!imageUrl} onOpenChange={onClose}>
      <DialogContent className="p-0 border border-white/20 bg-[#0a1017] shadow-2xl max-w-[90vw] md:max-w-[800px]">
        <DialogTitle className="sr-only">Satellite Image Preview</DialogTitle>
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.96 }}
          transition={{ duration: 0.2, ease: 'easeOut' }}
          className="relative bg-[#070b10] rounded-2xl overflow-hidden p-2"
        >
          <img
            src={imageUrl}
            alt="Full satellite preview"
            className="w-full max-h-[80vh] object-contain rounded-xl border border-white/10"
          />
        </motion.div>
      </DialogContent>
    </Dialog>
  )
}

// PromptInput Context and Components
interface PromptInputContextType {
  isLoading: boolean
  value: string
  setValue: (value: string) => void
  maxHeight: number | string
  onSubmit?: () => void
  disabled?: boolean
}

const PromptInputContext = React.createContext<PromptInputContextType>({
  isLoading: false,
  value: '',
  setValue: () => {},
  maxHeight: 240,
  onSubmit: undefined,
  disabled: false,
})

export function usePromptInput() {
  const context = React.useContext(PromptInputContext)
  if (!context) throw new Error('usePromptInput must be used within a PromptInput')
  return context
}

export interface PromptInputProps {
  isLoading?: boolean
  value?: string
  onValueChange?: (value: string) => void
  maxHeight?: number | string
  onSubmit?: () => void
  children: React.ReactNode
  className?: string
  disabled?: boolean
  onDragOver?: (e: React.DragEvent) => void
  onDragLeave?: (e: React.DragEvent) => void
  onDrop?: (e: React.DragEvent) => void
}

export const PromptInput = React.forwardRef<HTMLDivElement, PromptInputProps>(
  (
    {
      className,
      isLoading = false,
      maxHeight = 240,
      value,
      onValueChange,
      onSubmit,
      children,
      disabled = false,
      onDragOver,
      onDragLeave,
      onDrop,
    },
    ref
  ) => {
    const [internalValue, setInternalValue] = React.useState(value || '')

    React.useEffect(() => {
      if (value !== undefined) {
        setInternalValue(value)
      }
    }, [value])

    const handleChange = (newValue: string) => {
      setInternalValue(newValue)
      onValueChange?.(newValue)
    }

    return (
      <TooltipProvider>
        <PromptInputContext.Provider
          value={{
            isLoading,
            value: value !== undefined ? value : internalValue,
            setValue: handleChange,
            maxHeight,
            onSubmit,
            disabled,
          }}
        >
          <div
            ref={ref}
            className={cn(
              'rounded-2xl border border-white/15 bg-[#05080b] p-2.5 sm:p-3 shadow-2xl transition-all duration-200 focus-within:border-white/35 hover:border-white/25',
              isLoading && 'border-white/50 shadow-[0_0_20px_rgba(255,255,255,0.1)]',
              className
            )}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            {children}
          </div>
        </PromptInputContext.Provider>
      </TooltipProvider>
    )
  }
)
PromptInput.displayName = 'PromptInput'

export interface PromptInputTextareaProps {
  disableAutosize?: boolean
  placeholder?: string
}

export const PromptInputTextarea: React.FC<PromptInputTextareaProps & React.ComponentProps<typeof Textarea>> = ({
  className,
  onKeyDown,
  disableAutosize = false,
  placeholder,
  ...props
}) => {
  const { value, setValue, maxHeight, onSubmit, disabled } = usePromptInput()
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)

  React.useEffect(() => {
    if (disableAutosize || !textareaRef.current) return
    textareaRef.current.style.height = 'auto'
    textareaRef.current.style.height =
      typeof maxHeight === 'number'
        ? `${Math.min(textareaRef.current.scrollHeight, maxHeight)}px`
        : `min(${textareaRef.current.scrollHeight}px, ${maxHeight})`
  }, [value, maxHeight, disableAutosize])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSubmit?.()
    }
    onKeyDown?.(e)
  }

  return (
    <Textarea
      ref={textareaRef}
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      className={cn('text-sm sm:text-base text-white placeholder:text-white/40', className)}
      disabled={disabled}
      placeholder={placeholder}
      {...props}
    />
  )
}

export interface PromptInputActionsProps extends React.HTMLAttributes<HTMLDivElement> {}

export const PromptInputActions: React.FC<PromptInputActionsProps> = ({ children, className, ...props }) => (
  <div className={cn('flex items-center gap-2', className)} {...props}>
    {children}
  </div>
)

export interface PromptInputActionProps extends React.ComponentProps<typeof Tooltip> {
  tooltip: React.ReactNode
  children: React.ReactNode
  side?: 'top' | 'bottom' | 'left' | 'right'
  className?: string
}

export const PromptInputAction: React.FC<PromptInputActionProps> = ({
  tooltip,
  children,
  className,
  side = 'top',
  ...props
}) => {
  const { disabled } = usePromptInput()
  return (
    <Tooltip {...props}>
      <TooltipTrigger asChild disabled={disabled}>
        {children}
      </TooltipTrigger>
      <TooltipContent side={side} className={className}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  )
}

// Custom Divider Component
export const CustomDivider: React.FC = () => (
  <div className="relative h-5 sm:h-6 w-[1.5px] mx-0.5 sm:mx-1">
    <div
      className="absolute inset-0 bg-gradient-to-t from-transparent via-[#8eb7ff]/60 to-transparent rounded-full"
      style={{
        clipPath: 'polygon(0% 0%, 100% 0%, 100% 40%, 140% 50%, 100% 60%, 100% 100%, 0% 100%, 0% 60%, -40% 50%, 0% 40%)',
      }}
    />
  </div>
)

// Main PromptInputBox Component
export interface PromptInputBoxProps {
  onSend?: (message: string, files?: File[]) => void
  isLoading?: boolean
  placeholder?: string
  className?: string
  value?: string
  onValueChange?: (val: string) => void
}

export const PromptInputBox = React.forwardRef((props: PromptInputBoxProps, ref: React.Ref<HTMLDivElement>) => {
  const {
    onSend = () => {},
    isLoading = false,
    placeholder = 'Ask a question about this satellite tile...',
    className,
    value: controlledValue,
    onValueChange,
  } = props

  const [internalInput, setInternalInput] = React.useState('')
  const input = controlledValue !== undefined ? controlledValue : internalInput

  const setInput = (val: string) => {
    if (controlledValue === undefined) {
      setInternalInput(val)
    }
    onValueChange?.(val)
  }

  const [files, setFiles] = React.useState<File[]>([])
  const [filePreviews, setFilePreviews] = React.useState<{ [key: string]: string }>({})
  const [selectedImage, setSelectedImage] = React.useState<string | null>(null)
  const [isRecording, setIsRecording] = React.useState(false)
  const [showSearch, setShowSearch] = React.useState(false)
  const [showThink, setShowThink] = React.useState(false)
  const [showCanvas, setShowCanvas] = React.useState(false)
  const uploadInputRef = React.useRef<HTMLInputElement>(null)
  const promptBoxRef = React.useRef<HTMLDivElement>(null)

  const handleToggleChange = (mode: string) => {
    if (mode === 'search') {
      setShowSearch((prev) => !prev)
      setShowThink(false)
    } else if (mode === 'think') {
      setShowThink((prev) => !prev)
      setShowSearch(false)
    }
  }

  const handleCanvasToggle = () => setShowCanvas((prev) => !prev)

  const isImageFile = (file: File) => file.type.startsWith('image/')

  const processFile = (file: File) => {
    if (!isImageFile(file)) {
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      return
    }
    setFiles([file])
    const reader = new FileReader()
    reader.onload = (e) => setFilePreviews({ [file.name]: e.target?.result as string })
    reader.readAsDataURL(file)
  }

  const handleDragOver = React.useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDragLeave = React.useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = React.useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    const droppedFiles = Array.from(e.dataTransfer.files)
    const imageFiles = droppedFiles.filter((file) => isImageFile(file))
    if (imageFiles.length > 0) processFile(imageFiles[0])
  }, [])

  const handleRemoveFile = (index: number) => {
    const fileToRemove = files[index]
    if (fileToRemove && filePreviews[fileToRemove.name]) setFilePreviews({})
    setFiles([])
  }

  const openImageModal = (imageUrl: string) => setSelectedImage(imageUrl)

  const handlePaste = React.useCallback((e: ClipboardEvent) => {
    const items = e.clipboardData?.items
    if (!items) return
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile()
        if (file) {
          e.preventDefault()
          processFile(file)
          break
        }
      }
    }
  }, [])

  React.useEffect(() => {
    document.addEventListener('paste', handlePaste)
    return () => document.removeEventListener('paste', handlePaste)
  }, [handlePaste])

  const handleSubmit = () => {
    if (input.trim() || files.length > 0) {
      let messagePrefix = ''
      if (showSearch) messagePrefix = '[Earth Catalog: '
      else if (showThink) messagePrefix = '[RS-CLIP Deep Reason: '
      else if (showCanvas) messagePrefix = '[Spatial Grounding: '
      const formattedInput = messagePrefix ? `${messagePrefix}${input}]` : input
      onSend(formattedInput, files)
      setInput('')
      setFiles([])
      setFilePreviews({})
    }
  }

  const handleStartRecording = () => {}

  const handleStopRecording = (duration: number) => {
    setIsRecording(false)
    onSend(`[Voice audio query - ${duration}s]`, [])
  }

  const hasContent = input.trim() !== '' || files.length > 0

  return (
    <>
      <PromptInput
        value={input}
        onValueChange={setInput}
        isLoading={isLoading}
        onSubmit={handleSubmit}
        className={cn(
          'w-full bg-[#070b10]/95 border-white/15 shadow-[0_12px_40px_rgba(0,0,0,0.6)] transition-all duration-300 ease-in-out',
          isRecording && 'border-red-500/70 shadow-[0_0_30px_rgba(239,68,68,0.25)]',
          className
        )}
        disabled={isLoading || isRecording}
        ref={ref || promptBoxRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {files.length > 0 && !isRecording && (
          <div className="flex flex-wrap gap-2 p-0 pb-2 transition-all duration-300">
            {files.map((file, index) => (
              <div key={index} className="relative group">
                {file.type.startsWith('image/') && filePreviews[file.name] && (
                  <div
                    className="w-16 h-16 sm:w-20 sm:h-20 rounded-xl overflow-hidden cursor-pointer border border-white/20 transition-all duration-300 hover:border-[#8eb7ff]"
                    onClick={() => openImageModal(filePreviews[file.name])}
                  >
                    <img
                      src={filePreviews[file.name]}
                      alt={file.name}
                      className="h-full w-full object-cover"
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleRemoveFile(index)
                      }}
                      className="absolute top-1 right-1 rounded-full bg-black/80 p-1 text-white hover:bg-red-500 transition-colors"
                      title="Remove satellite tile"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <div
          className={cn(
            'transition-all duration-300',
            isRecording ? 'h-0 overflow-hidden opacity-0' : 'opacity-100'
          )}
        >
          <PromptInputTextarea
            placeholder={
              showSearch
                ? 'Search global satellite catalog (Sentinel-1 SAR, Sentinel-2 Optical, Landsat)...'
                : showThink
                ? 'Deep RS-CLIP multi-modal reasoning & land-cover analysis...'
                : showCanvas
                ? 'Generate GeoJSON coordinate polygons and spatial segmentation...'
                : placeholder
            }
            className="text-sm sm:text-base text-white placeholder:text-white/40"
          />
        </div>

        {isRecording && (
          <VoiceRecorder
            isRecording={isRecording}
            onStartRecording={handleStartRecording}
            onStopRecording={handleStopRecording}
          />
        )}

        <PromptInputActions className="flex items-center justify-between gap-1.5 sm:gap-2 p-0 pt-2">
          <div
            className={cn(
              'flex items-center gap-1 sm:gap-1.5 transition-opacity duration-300 overflow-x-auto scrollbar-none',
              isRecording ? 'opacity-0 invisible h-0' : 'opacity-100 visible'
            )}
          >
            <PromptInputAction tooltip="Upload satellite imagery tile (Optical or SAR)">
              <button
                type="button"
                onClick={() => uploadInputRef.current?.click()}
                className="flex h-8 w-8 text-white/50 cursor-pointer items-center justify-center rounded-full transition-colors hover:bg-white/10 hover:text-white"
                disabled={isRecording}
                aria-label="Upload satellite tile"
              >
                <Paperclip className="h-4 w-4 transition-colors" />
                <input
                  ref={uploadInputRef}
                  type="file"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) processFile(e.target.files[0])
                    if (e.target) e.target.value = ''
                  }}
                  accept="image/*,.tif,.tiff"
                />
              </button>
            </PromptInputAction>

            <div className="flex items-center">
              <button
                type="button"
                onClick={() => handleToggleChange('search')}
                className={cn(
                  'rounded-full transition-all flex items-center gap-1.5 px-2.5 py-1 border h-7 sm:h-8 text-xs cursor-pointer font-medium',
                  showSearch
                    ? 'bg-white/15 border-white/30 text-white'
                    : 'bg-transparent border-transparent text-white/45 hover:text-white hover:bg-white/5'
                )}
                title="Earth Catalog Search"
              >
                <div className="w-3.5 h-3.5 flex items-center justify-center flex-shrink-0">
                  <motion.div
                    animate={{ rotate: showSearch ? 360 : 0, scale: showSearch ? 1.05 : 1 }}
                    whileHover={{ rotate: showSearch ? 360 : 15, scale: 1.05, transition: { type: 'spring', stiffness: 300, damping: 10 } }}
                    transition={{ type: 'spring', stiffness: 260, damping: 25 }}
                  >
                    <Globe className={cn('w-3.5 h-3.5', showSearch ? 'text-white' : 'text-inherit')} />
                  </motion.div>
                </div>
                <AnimatePresence>
                  {showSearch && (
                    <motion.span
                      initial={{ width: 0, opacity: 0 }}
                      animate={{ width: 'auto', opacity: 1 }}
                      exit={{ width: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="text-[11px] overflow-hidden whitespace-nowrap text-white flex-shrink-0"
                    >
                      Catalog
                    </motion.span>
                  )}
                </AnimatePresence>
              </button>

              <CustomDivider />

              <button
                type="button"
                onClick={() => handleToggleChange('think')}
                className={cn(
                  'rounded-full transition-all flex items-center gap-1.5 px-2.5 py-1 border h-7 sm:h-8 text-xs cursor-pointer font-medium',
                  showThink
                    ? 'bg-white/15 border-white/30 text-white'
                    : 'bg-transparent border-transparent text-white/45 hover:text-white hover:bg-white/5'
                )}
                title="Deep Multimodal Reasoning"
              >
                <div className="w-3.5 h-3.5 flex items-center justify-center flex-shrink-0">
                  <motion.div
                    animate={{ rotate: showThink ? 360 : 0, scale: showThink ? 1.05 : 1 }}
                    whileHover={{ rotate: showThink ? 360 : 15, scale: 1.05, transition: { type: 'spring', stiffness: 300, damping: 10 } }}
                    transition={{ type: 'spring', stiffness: 260, damping: 25 }}
                  >
                    <BrainCog className={cn('w-3.5 h-3.5', showThink ? 'text-white' : 'text-inherit')} />
                  </motion.div>
                </div>
                <AnimatePresence>
                  {showThink && (
                    <motion.span
                      initial={{ width: 0, opacity: 0 }}
                      animate={{ width: 'auto', opacity: 1 }}
                      exit={{ width: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="text-[11px] overflow-hidden whitespace-nowrap text-white flex-shrink-0"
                    >
                      Reason
                    </motion.span>
                  )}
                </AnimatePresence>
              </button>

              <CustomDivider />

              <button
                type="button"
                onClick={handleCanvasToggle}
                className={cn(
                  'rounded-full transition-all flex items-center gap-1.5 px-2.5 py-1 border h-7 sm:h-8 text-xs cursor-pointer font-medium',
                  showCanvas
                    ? 'bg-white/15 border-white/30 text-white'
                    : 'bg-transparent border-transparent text-white/45 hover:text-white hover:bg-white/5'
                )}
                title="Spatial Grounding Map & GeoJSON"
              >
                <div className="w-3.5 h-3.5 flex items-center justify-center flex-shrink-0">
                  <motion.div
                    animate={{ rotate: showCanvas ? 360 : 0, scale: showCanvas ? 1.05 : 1 }}
                    whileHover={{ rotate: showCanvas ? 360 : 15, scale: 1.05, transition: { type: 'spring', stiffness: 300, damping: 10 } }}
                    transition={{ type: 'spring', stiffness: 260, damping: 25 }}
                  >
                    <FolderCode className={cn('w-3.5 h-3.5', showCanvas ? 'text-white' : 'text-inherit')} />
                  </motion.div>
                </div>
                <AnimatePresence>
                  {showCanvas && (
                    <motion.span
                      initial={{ width: 0, opacity: 0 }}
                      animate={{ width: 'auto', opacity: 1 }}
                      exit={{ width: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="text-[11px] overflow-hidden whitespace-nowrap text-white flex-shrink-0"
                    >
                      Ground
                    </motion.span>
                  )}
                </AnimatePresence>
              </button>
            </div>
          </div>

          <PromptInputAction
            tooltip={
              isLoading
                ? 'Processing remote sensing query...'
                : isRecording
                ? 'Stop voice recording'
                : hasContent
                ? 'Send query to SatQuery engine'
                : 'Click to ask via voice'
            }
          >
            <Button
              variant="default"
              size="icon"
              className={cn(
                'h-8 w-8 sm:h-9 sm:w-9 rounded-full transition-all duration-200',
                isRecording
                  ? 'bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/40'
                  : hasContent
                  ? 'bg-white hover:bg-white/85 text-black font-semibold'
                  : 'bg-transparent hover:bg-white/10 text-white/45 hover:text-white'
              )}
              onClick={() => {
                if (isRecording) setIsRecording(false)
                else if (hasContent) handleSubmit()
                else setIsRecording(true)
              }}
              disabled={isLoading && !hasContent}
              aria-label={hasContent ? 'Send message' : isRecording ? 'Stop recording' : 'Voice input'}
            >
              {isLoading ? (
                <Square className="h-3.5 w-3.5 fill-current animate-pulse text-black" />
              ) : isRecording ? (
                <StopCircle className="h-4 w-4 text-red-400" />
              ) : hasContent ? (
                <ArrowUp className="h-4 w-4 text-black stroke-[2.5]" />
              ) : (
                <Mic className="h-4 w-4 transition-colors" />
              )}
            </Button>
          </PromptInputAction>
        </PromptInputActions>
      </PromptInput>

      <ImageViewDialog imageUrl={selectedImage} onClose={() => setSelectedImage(null)} />
    </>
  )
})
PromptInputBox.displayName = 'PromptInputBox'

// Standalone Demo component exported as requested in the snippet
export const DemoOne: React.FC = () => {
  return (
    <div className="flex w-full min-h-[400px] justify-center items-center bg-[#05080b] p-4">
      <div className="p-4 w-full max-w-[640px]">
        <PromptInputBox
          onSend={(message, files) => {
            console.log('Query sent:', message, files)
          }}
        />
      </div>
    </div>
  )
}
