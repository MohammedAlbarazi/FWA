import "./globals.css";

export const metadata = {
  title: "FWA Investigation Dashboard",
  description: "Risk-scored health insurance claims for investigator review (synthetic data)",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
