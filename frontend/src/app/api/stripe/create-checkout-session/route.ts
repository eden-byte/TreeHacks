import Stripe from "stripe";
import { NextResponse } from "next/server";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY as string, {
  apiVersion: "2022-11-15",
});

export async function POST(req: Request) {
  try {
    const { tier } = await req.json();

    const priceId = process.env.STRIPE_PRICE_PREMIUM;
    if (!priceId) {
      return NextResponse.json({ error: "STRIPE_PRICE_PREMIUM not configured" }, { status: 500 });
    }

    const origin = process.env.NEXT_PUBLIC_BASE_URL ?? "http://localhost:3000";

    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      payment_method_types: ["card"],
      line_items: [{ price: priceId, quantity: 1 }],
      success_url: `${origin}/account/billing/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${origin}/account/billing/cancel`,
    });

    return NextResponse.json({ url: session.url });
  } catch (err) {
    console.error("/api/stripe/create-checkout-session error", err);
    return NextResponse.json({ error: "Unable to create checkout session" }, { status: 500 });
  }
}
