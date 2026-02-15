# Vera Web Application

An elegant, accessible web interface for Vera - an AI-powered assistive system for visually impaired users.

## 🌟 Features

### User Interface (Accessibility-First)
- **Dashboard**: Large, high-contrast cards with full keyboard navigation
- **My Research**: Timeline of deep research tasks with export functionality
- **Saved Products**: Grid view of identified products with filtering
- **Memory Management**: Timeline view with bulk delete actions
- **Account & Pricing**: Subscription management and settings

### Healthcare Provider Interface
- **Live Feed**: Real-time camera view with object detection overlays
- **Motor Visualization**: 5-motor haptic necklace visualization with pulsing animations
- **System Status**: Jetson Nano metrics and feature monitoring

## 🎯 Accessibility Features

- **WCAG AAA Compliant**: 7:1 color contrast for normal text
- **Full Keyboard Navigation**: Arrow keys, Enter, Tab, Escape
- **Screen Reader Optimized**: Semantic HTML, ARIA labels, skip links
- **Large Text**: Minimum 16px body text, scalable to 200%
- **Focus Indicators**: 4px solid borders with high contrast
- **Touch Targets**: Minimum 44x44px for all interactive elements

## 🚀 Getting Started

### Prerequisites
- Node.js 18.x or higher (recommended: 20.x)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser to [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
npm start
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/                      # Next.js 14 App Router
│   │   ├── (user)/              # User interface routes
│   │   │   ├── dashboard/
│   │   │   ├── research/
│   │   │   ├── products/
│   │   │   ├── memory/
│   │   │   ├── account/
│   │   │   └── layout.tsx
│   │   ├── (provider)/          # Healthcare provider routes
│   │   │   ├── live-feed/
│   │   │   ├── motor-visualization/
│   │   │   ├── system-status/
│   │   │   └── layout.tsx
│   │   ├── layout.tsx           # Root layout
│   │   └── page.tsx             # Landing page
│   ├── components/              # Reusable components
│   ├── lib/                     # Utilities and hooks
│   └── styles/
│       └── globals.css          # Global styles with Tailwind
├── public/                      # Static assets
├── tailwind.config.ts           # Tailwind configuration
├── next.config.js               # Next.js configuration
└── package.json
```

## 🎨 Design Tokens

### Colors
- **Primary**: `#1e3a8a` (Deep Blue)
- **Secondary**: `#10b981` (Green)
- **Accent**: `#f59e0b` (Amber)
- **Background**: `#fafafa` (Warm White)
- **Text Primary**: `#1f2937` (Dark Gray)
- **Focus**: `#3b82f6` (Blue)

### Typography
- **Font Family**: Inter (Google Fonts)
- **Minimum Size**: 16px body text
- **Headings**: 700 weight
- **Body**: 400 weight

## 🛠️ Tech Stack

- **Framework**: Next.js 14 (App Router) with TypeScript
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Forms**: React Hook Form + Zod (ready for integration)
- **Real-time**: Socket.io client (prepared for WebSocket)

## 🌐 Deployment to Vercel

### Option 1: Vercel CLI

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Deploy from the frontend directory:
```bash
cd frontend
vercel
```

3. Follow the prompts to link your project

4. For production deployment:
```bash
vercel --prod
```

### Option 2: Vercel Dashboard

1. Push your code to GitHub
2. Visit [vercel.com](https://vercel.com)
3. Click "Import Project"
4. Select your repository
5. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
6. Click "Deploy"

### Environment Variables (if needed)

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=https://your-api-url.com
NEXT_PUBLIC_WS_URL=wss://your-websocket-url.com
```

Then add these in Vercel Dashboard → Settings → Environment Variables

## 🎹 Keyboard Shortcuts

### Global
- **Tab**: Navigate between interactive elements
- **Shift + Tab**: Navigate backwards
- **Enter**: Activate/select
- **Escape**: Go back/close

### User Interface
- **↑/↓ Arrow Keys**: Navigate menu items
- **Enter**: Open selected page
- **Escape**: Return to home

## 📱 Responsive Design

- **Mobile**: 320px - 767px (single column, bottom nav)
- **Tablet**: 768px - 1023px (two-column grid)
- **Desktop**: 1024px+ (full layout with sidebar)

## 🔄 Future Integration

This frontend is designed to integrate with:
- Python FastAPI backend running on Jetson Nano
- WebSocket for real-time motor data and live feed
- JWT authentication
- Stripe payments for subscriptions
- Cloud storage for user data

## 🧪 Testing Accessibility

### Browser DevTools
1. Open DevTools (F12)
2. Go to Lighthouse tab
3. Run accessibility audit
4. Target score: 100

### Keyboard Navigation Test
1. Disconnect your mouse
2. Use only keyboard (Tab, Arrow keys, Enter, Esc)
3. Verify all features are accessible

### Screen Reader Test
- **Windows**: NVDA or JAWS
- **macOS**: VoiceOver (Cmd + F5)
- **Linux**: Orca

### Color Contrast
Use browser extensions:
- WAVE Evaluation Tool
- axe DevTools
- Color Contrast Analyzer

## 📊 Performance

Target Lighthouse scores:
- **Performance**: >90
- **Accessibility**: 100
- **Best Practices**: >95
- **SEO**: >90

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

This is a TreeHacks 2026 project. For contributions or issues, please contact the Vera team.

## 📞 Support

For questions about this web application, please refer to:
- Project documentation
- Code comments
- Component descriptions

---

**Built with ❤️ for TreeHacks 2026**
