"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { feedbackApi } from "@/api/client";
import { ThumbsUp, ThumbsDown } from "lucide-react";

interface Props {
  queryId: string;
  open: boolean;
  onClose: () => void;
}

export function FeedbackDialog({ queryId, open, onClose }: Props) {
  const [rating, setRating] = useState<"positive" | "negative">("positive");
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      await feedbackApi.submit({
        query_id: queryId,
        rating: rating === "positive" ? 5 : 1,
        comment: comment || undefined,
      });
      onClose();
      setComment("");
    } catch {
      // silent
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>How was this response?</DialogTitle>
          <DialogDescription>
            Your feedback helps us improve the quality of answers.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="flex gap-3">
            <Button
              variant={rating === "positive" ? "default" : "outline"}
              className="flex-1"
              onClick={() => setRating("positive")}
            >
              <ThumbsUp className="h-4 w-4 mr-2" />
              Helpful
            </Button>
            <Button
              variant={rating === "negative" ? "default" : "outline"}
              className="flex-1"
              onClick={() => setRating("negative")}
            >
              <ThumbsDown className="h-4 w-4 mr-2" />
              Not helpful
            </Button>
          </div>

          <Textarea
            placeholder="Optional: tell us how we can improve..."
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={3}
          />
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={submitting}>
            {submitting ? "Submitting..." : "Submit"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
