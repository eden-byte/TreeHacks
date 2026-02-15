import Stripe from "stripe";
import { NextResponse } from "next/server";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY as string, {
  apiVersion: "2022-11-15",
});

export async function POST(req: Request) {
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!webhookSecret) return new Response("Webhook secret not configured", { status: 500 });

  const sig = req.headers.get("stripe-signature") ?? "";
  const buf = Buffer.from(await req.arrayBuffer());

  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(buf, sig, webhookSecret);
  } catch (err) {
    console.error("Stripe webhook signature verification failed:", err);
    return new Response("Invalid signature", { status: 400 });
  }

  // Basic event handling (extend for your app)
  switch (event.type) {
    case "checkout.session.completed":
      console.log("checkout.session.completed", (event.data.object as any).id);
      // TODO: mark user subscription active in DB
      break;
    case "invoice.payment_succeeded":
      console.log("invoice.payment_succeeded", (event.data.object as any).id);
      break;
    case "customer.subscription.updated":
    case "customer.subscription.deleted":
      console.log(event.type, (event.data.object as any).id);
      break;
    default:
      console.log(`Unhandled Stripe event: ${event.type}`);
  }

  return NextResponse.json({ received: true });
}
